"""Phase 3: deep station model (SeamNet).

    python train.py --campaign NAME [--steps 250] [--folds 5] [--final]       grouped CV inside a campaign
    python train.py --train A B --test C [--steps 2000]                       train on A+B, evaluate on C
    common: [--device cpu|cuda] [--workers 4]

SeamNet: one pretrained backbone (4 input channels: RGB + signed distance to the seam) applied to
both scales of a station's crop; pooled features are concatenated and fed to two heads:
  error   seam error yes/no (BCE)
  type    batiment_coupe / vehicule_coupe / decalage / radiometrie (CE, positives only)

Each batch mixes real positives, synthetic positives made from clean training stations (synth.py)
and negatives; batches are built by DataLoader workers, reading crops from the cache when a
campaign has one (dataset.py) and from its mosaic otherwise.

CV mode writes out/<campaign>/cnn_scores.parquet (out-of-fold) and appends to baseline_report.md;
--final also saves out/<campaign>/seamnet.pt. Transfer mode saves SeamNet and a GBM trained on A+B
to out/model_A+B/ (usable as infer.py --models-from model_A+B) and writes the evaluation on C to
out/<C>/transfer_from_A+B.md, including the operating points for automatic flagging.
"""

import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import rankdata
from sklearn.model_selection import GroupKFold

import synth
from baseline import CONTEXT, fit_gbm, tile_groups
from config import OUT, load_campaigns
from dataset import Source, flip, iter_crops, jitter, to_tensor
from metrics import evaluate, operating_points, targets, threshold_table


class SeamNet(nn.Module):
    def __init__(self, backbone="resnet18", pretrained=True, n_types=len(synth.TYPES)):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, in_chans=4, num_classes=0)
        d = self.backbone.num_features
        self.neck = nn.Sequential(nn.Linear(2 * d, 256), nn.ReLU(), nn.Dropout(0.3))
        self.error = nn.Linear(256, 1)
        self.kind = nn.Linear(256, n_types)  # not "type": nn.Module.type() exists

    def forward(self, x):
        """x: (B, 2 scales, 4, H, W) -> (error logit (B,), type logits (B, n_types))"""
        b, s = x.shape[:2]
        f = self.backbone(x.flatten(0, 1)).view(b, -1)
        h = self.neck(f)
        return self.error(h).squeeze(1), self.kind(h)


def pools(sources, stations, idx_per_source):
    """Training positives (source, station, type class) and negatives (source, station)."""
    pos, neg = [], []
    for s, (st, idx) in enumerate(zip(stations, idx_per_source)):
        lab = st["label"].to_numpy()[idx]
        typ = st["seam_type"].to_numpy()[idx]
        pos += [(s, i, synth.REAL_TYPE.get(int(t), -100)) for i, l, t in zip(idx, lab, typ) if l == 1]
        neg += [(s, i) for i, l in zip(idx, lab) if l == 0]
    return np.array(pos, int).reshape(-1, 3), np.array(neg, int).reshape(-1, 2)


def batch(sources, pos, neg, rng, n_pos=8, n_syn=8, n_neg=16):
    xs, ys, ts = [], [], []
    for s, i, t in pos[rng.integers(len(pos), size=n_pos)]:
        xs.append(jitter(flip(sources[s][i], rng), rng)); ys.append(1); ts.append(t)
    for s, i in neg[rng.integers(len(neg), size=n_syn)]:
        c, t = synth.make(sources[s][i], rng)
        xs.append(jitter(flip(c, rng), rng)); ys.append(1); ts.append(t)
    for s, i in neg[rng.integers(len(neg), size=n_neg)]:
        xs.append(jitter(flip(sources[s][i], rng), rng)); ys.append(0); ts.append(-100)
    return (torch.stack([to_tensor(x) for x in xs]), torch.tensor(ys, dtype=torch.float32),
            torch.tensor(ts))


class BatchStream(torch.utils.data.IterableDataset):
    """`steps` training batches, split between the DataLoader workers, each with its own RNG."""

    def __init__(self, sources, pos, neg, steps, seed):
        self.sources, self.pos, self.neg, self.steps, self.seed = sources, pos, neg, steps, seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        rng = np.random.default_rng([self.seed, wid])
        for _ in range(wid, self.steps, nw):
            yield batch(self.sources, self.pos, self.neg, rng)


def fit(sources, pos, neg, steps, device, seed=0, workers=4):
    torch.manual_seed(seed)
    model = SeamNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=steps, pct_start=0.1)
    loader = torch.utils.data.DataLoader(BatchStream(sources, pos, neg, steps, seed), batch_size=None,
                                         num_workers=workers, prefetch_factor=2 if workers else None)
    model.train()
    t0 = time.time()
    for step, (x, y, t) in enumerate(loader):
        x, y, t = x.to(device), y.to(device), t.to(device)
        err, typ = model(x)
        loss = F.binary_cross_entropy_with_logits(err, y)
        if (t >= 0).any():
            loss = loss + 0.5 * F.cross_entropy(typ, t, ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if (step + 1) % 50 == 0:
            print(f"    step {step + 1}/{steps} loss {loss.item():.3f} ({time.time() - t0:.0f}s)", flush=True)
    return model


@torch.no_grad()
def predict(model, name, idx, device, workers=8):
    """Error probability and type probabilities, averaged over the identity and an along-seam flip."""
    model.eval()
    err, typ = [], []
    for c in iter_crops(name, np.asarray(idx), workers):
        x = torch.stack([to_tensor(a) for a in c]).to(device)
        e1, t1 = model(x)
        e2, t2 = model(torch.flip(x, dims=[3]))
        err.append(((torch.sigmoid(e1) + torch.sigmoid(e2)) / 2).cpu().numpy())
        typ.append(((F.softmax(t1, 1) + F.softmax(t2, 1)) / 2).cpu().numpy())
    return np.concatenate(err), np.concatenate(typ)


def report_table(st, scores, pts):
    return pd.DataFrame({f"{k}{tag}": evaluate(st, s, pts, on_seam)
                         for on_seam, tag in ((False, ""), (True, " (on-seam)")) for k, s in scores.items()}).T


def run_cv(name, steps, folds, final, device, workers):
    out = OUT / name
    st = pd.read_parquet(out / "stations.parquet")
    src = [Source(name)]
    groups = tile_groups(st)
    pts = targets(out / "artefacts.gpkg")
    score = np.full(len(st), np.nan)
    types = np.full((len(st), len(synth.TYPES)), np.nan)
    for k, (tr, te) in enumerate(GroupKFold(folds).split(st, groups=groups)):
        print(f"fold {k + 1}/{folds}: {len(tr)} train, {len(te)} test", flush=True)
        model = fit(src, *pools(src, [st], [tr]), steps, device, seed=k, workers=workers)
        score[te], types[te] = predict(model, name, te, device)
    df = pd.DataFrame(types, columns=[f"p_{t}" for t in synth.TYPES], index=st.index)
    df.insert(0, "cnn", score)
    df.to_parquet(out / "cnn_scores.parquet")
    # expected precision/recall per threshold, read by detect.py
    (out / "thresholds.json").write_text(json.dumps({"campaign": name, "source": "SeamNet out-of-fold",
                                                     "rows": threshold_table(st, score, pts)}, indent=1))

    scores = {"cnn": score}
    base = out / "baseline_scores.parquet"
    if base.exists():
        g = pd.read_parquet(base)["gbm_all"].to_numpy()
        scores["gbm_all"] = g
        scores["cnn+gbm (rank avg)"] = rankdata(score) + rankdata(g)
    table = report_table(st, scores, pts)
    print(table.drop(columns=["km"]).to_string())
    with open(out / "baseline_report.md", "a") as f:
        f.write(f"\n## SeamNet ({steps} steps/fold, {folds}-fold grouped CV)\n\n"
                f"{table.drop(columns=['km']).to_markdown()}\n")
    if final:
        print("final model on all stations", flush=True)
        model = fit(src, *pools(src, [st], [np.arange(len(st))]), steps, device, seed=99, workers=workers)
        torch.save(model.state_dict(), out / "seamnet.pt")


def run_transfer(train, test, steps, device, workers):
    tag = "+".join(train)
    mdir = OUT / f"model_{tag}"
    mdir.mkdir(parents=True, exist_ok=True)
    stations = [pd.read_parquet(OUT / n / "stations.parquet") for n in train]
    src = [Source(n) for n in train]
    pos, neg = pools(src, stations, [np.arange(len(s)) for s in stations])
    print(f"train on {tag}: {len(pos)} positive, {len(neg)} negative stations", flush=True)
    model = fit(src, pos, neg, steps, device, seed=0, workers=workers)
    torch.save(model.state_dict(), mdir / "seamnet.pt")
    # feature column order depends on the first rows of each campaign: align by name
    Xs = [pd.concat([s[CONTEXT], pd.read_parquet(OUT / n / "features.parquet")], axis=1)
          for n, s in zip(train, stations)]
    X = pd.concat([x[Xs[0].columns] for x in Xs], ignore_index=True)
    y = np.concatenate([s["label"].to_numpy() for s in stations])
    g = fit_gbm(X[y >= 0], y[y >= 0])
    joblib.dump(g, mdir / "gbm_all.joblib")

    out = OUT / test
    st = pd.read_parquet(out / "stations.parquet")
    pts = targets(out / "artefacts.gpkg")
    s_cnn, s_typ = predict(model, test, np.arange(len(st)), device)
    Xt = pd.concat([st[CONTEXT], pd.read_parquet(out / "features.parquet")], axis=1)
    s_gbm = g.predict_proba(Xt[g.feature_names_in_])[:, 1]
    scores = {"cnn": s_cnn, "gbm_all": s_gbm, "cnn+gbm (rank avg)": rankdata(s_cnn) + rankdata(s_gbm)}
    df = pd.DataFrame(s_typ, columns=[f"p_{t}" for t in synth.TYPES], index=st.index)
    df.insert(0, "cnn", s_cnn)
    df.insert(1, "gbm_all", s_gbm)
    df.to_parquet(out / f"transfer_from_{tag}.parquet")
    table = report_table(st, scores, pts)
    ops = operating_points(st, s_cnn, pts)
    print(table.drop(columns=["km"]).to_string())
    print(ops.to_string(index=False))
    (out / f"transfer_from_{tag}.md").write_text(
        f"# SeamNet trained on {tag}, evaluated on {test}\n\n{len(pos)} training positives, {steps} steps.\n\n"
        f"{table.drop(columns=['km']).to_markdown()}\n\n## SeamNet operating points (detections after NMS)\n\n"
        f"{ops.to_markdown(index=False)}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--campaign", help="grouped CV inside this campaign")
    ap.add_argument("--train", nargs="+", help="transfer mode: campaigns to train on")
    ap.add_argument("--test", help="transfer mode: campaign to evaluate on")
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=4, help="DataLoader workers building batches")
    args = ap.parse_args()
    if args.device == "cpu":
        torch.set_num_threads(max(1, 12 - args.workers))
    names = load_campaigns()
    if args.train:
        assert args.test and all(n in names for n in args.train + [args.test]), "unknown campaign"
        run_transfer(args.train, args.test, args.steps, args.device, args.workers)
    else:
        for name in names:
            if args.campaign in (None, name):
                run_cv(name, args.steps, args.folds, args.final, args.device, args.workers)


if __name__ == "__main__":
    main()
