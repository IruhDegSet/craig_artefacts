"""Hotspots (specular sun glint): find every glint, then decide which ones an operator would report.

    python hotspot.py --campaign NAME [--no-bdtopo] [--workers 8]          candidates, rule, classifier (grouped CV)
    python hotspot.py --train A B --test C [--min-positives 20]           classifier trained on A+B, evaluated on C

A CRAIG "hotspot" is a saturated white disc, often with star rays, on a roof (skylight, metal,
panel) or a windscreen (tuto_cv.pdf pp. 6, 19-22). It is reported only when visible at 1/800 and
disturbing; small glints on cars and private plots are not. Finding glints is easy; the task is
*reportability*:

  1. candidates   min(R,G,B) >= SAT pixels, merged within MERGE m, core >= MIN_AREA m2
  2. features     core size and shape, halo radius, bloom (glow beyond the core outline, the cue that
                  separates a glint from a white object), star rays, colour, contrast, BD TOPO context
  3. decision     a gradient-boosting classifier once a campaign has >= MIN_POSITIVES reported
                  hotspots (grouped CV, saved as hotspot_gbm.joblib), else the RULE below

For a labelled campaign, writes out/<campaign>/hotspot_candidates.parquet (features, label, decision)
and hotspot_report.md (precision/recall of the rule and, when trained, of the classifier).
Transfer mode reads those candidates (run --campaign first for A, B and C), saves the classifier to
out/model_A+B/ (next to SeamNet's, so detect.py --models out/model_A+B uses it) and writes
out/C/hotspot_transfer_from_A+B.md. The rule's thresholds were chosen on Eure-et-Loir, so on the
other campaigns the rule is itself a transfer result.
detect.py calls find_hotspots() on new tiles.
"""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import cv2
import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import rasterio
import shapely
from scipy.spatial import cKDTree

import bdtopo
from baseline import gbm, tile_groups
from config import OUT, load_campaigns

SAT = 245          # min(R,G,B) of a saturated glint pixel
MERGE = 0.5        # m: star rays and cores closer than this are one glint
MIN_AREA = 0.05    # m2 of saturated core
HALO_R = 6.0       # m: radius of the window used for the halo and rays
LABEL_RADIUS = 3.0  # m: a candidate this close to a hotspot point is that hotspot (clicks are approximate)
MIN_POSITIVES = 20  # reported hotspots needed before the classifier replaces the rule
HOTSPOT = 10       # art_arttype_id


def rule(c):
    """Operator criterion while labels are scarce: a big saturated core that glows (bloom) beyond its
    outline. White objects (roofs, tents, skylights, pools, vehicles) are as saturated but end at a sharp
    edge. On the 44 Eure-et-Loir tiles: 5/5 reported hotspots, 1 other flag out of 797 candidates;
    the two thresholds were chosen on those same 5, so this is an in-sample figure."""
    return ((c["core_m2"] >= 2.0) & (c["bloom_rel"] >= 0.4)).to_numpy()


def blob_features(gray, hsv, lab_id, labels, core, cx, cy, gsd):
    """Features of one candidate (pixel coordinates cx, cy in the tile)."""
    h, w = gray.shape
    r = int(HALO_R / gsd)
    y0, y1, x0, x1 = max(0, int(cy) - r), min(h, int(cy) + r + 1), max(0, int(cx) - r), min(w, int(cx) + r + 1)
    g = gray[y0:y1, x0:x1].astype(np.float32)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    rad = np.hypot(xx - cx, yy - cy) * gsd
    ang = np.arctan2(yy - cy, xx - cx)
    comp = labels[y0:y1, x0:x1] == lab_id
    sat_core = comp & core[y0:y1, x0:x1]
    core_m2 = sat_core.sum() * gsd * gsd
    core_r = np.sqrt(core_m2 / np.pi)

    cnt, _ = cv2.findContours(sat_core.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perim = sum(cv2.arcLength(c, True) for c in cnt) * gsd
    roundness = float(4 * np.pi * core_m2 / perim ** 2) if perim > 0 else 0.0

    bg = float(np.median(g[rad >= HALO_R - 1])) if (rad >= HALO_R - 1).any() else float(g.mean())
    bins = np.arange(0, HALO_R, 0.25)
    prof = np.array([g[(rad >= b) & (rad < b + 0.25)].mean() if ((rad >= b) & (rad < b + 0.25)).any() else bg
                     for b in bins])
    above = prof > bg + 0.3 * (255 - bg)
    halo_r = float(bins[np.argmin(above)] if not above.all() else HALO_R) if above[0] else 0.0

    ring = (rad > core_r + 0.3) & (rad < core_r + 2.0)
    star = 0.0
    if ring.any():
        d = np.abs(((ang + np.pi / 8) % (np.pi / 4)) - np.pi / 8)  # 0 on the 8 ray directions
        ray, gap = ring & (d < np.pi / 32), ring & (d > np.pi / 16)
        if ray.any() and gap.any():
            star = float(g[ray].mean() - g[gap].mean())

    halo = (rad > core_r) & (rad < max(core_r + 1.0, halo_r))
    # bloom: a glint glows beyond its core, a white object has a sharp edge. Measured at a distance
    # from the core outline (distance transform), so it does not depend on the core's shape.
    dcore = cv2.distanceTransform((~sat_core).astype(np.uint8), cv2.DIST_L2, 3) * gsd
    near, mid = (dcore > 0.25) & (dcore <= 0.75), (dcore > 0.75) & (dcore <= 1.5)
    bloom_near = float(g[near].mean() - bg) if near.any() else 0.0
    bloom_mid = float(g[mid].mean() - bg) if mid.any() else 0.0
    return {
        "core_m2": float(core_m2), "core_d_m": float(2 * core_r), "roundness": roundness,
        "halo_d_m": float(2 * halo_r), "halo_core_ratio": float(halo_r / core_r) if core_r > 0 else 0.0,
        "star": star, "bg": bg, "peak_contrast": float(255 - bg),
        "halo_sat": float(hsv[y0:y1, x0:x1, 1][halo].mean()) if halo.any() else 0.0,
        "core_frac_white": float((hsv[y0:y1, x0:x1, 1][sat_core] < 20).mean()) if sat_core.any() else 0.0,
        "local_std": float(g.std()),
        "bloom_near": bloom_near, "bloom_mid": bloom_mid, "bloom_rel": bloom_near / max(1.0, 255 - bg),
        "bloom_ratio": float(bloom_mid / bloom_near) if bloom_near > 1 else 0.0,
    }


def tile_candidates(path, use_bdtopo=True):
    """Glint candidates of one tile with their features, in EPSG:2154."""
    with rasterio.open(path) as ds:
        rgb = np.moveaxis(ds.read([1, 2, 3]), 0, -1)
        T, bounds, gsd = ds.transform, tuple(ds.bounds), abs(ds.transform.a)
        valid = ds.read(4) > 0 if ds.count >= 4 else np.ones(rgb.shape[:2], bool)
    core = (rgb.min(-1) >= SAT) & valid
    k = max(1, int(round(MERGE / gsd)))
    merged = cv2.dilate(core.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
    n, labels = cv2.connectedComponents(merged, connectivity=8)
    if n <= 1:
        return pd.DataFrame()
    core_px = np.bincount(labels[core], minlength=n)
    # position = centroid of the saturated core, not of the dilated (merged) blob
    ys, xs = np.nonzero(core)
    lc = labels[ys, xs]
    ccx = np.bincount(lc, weights=xs, minlength=n) / np.maximum(core_px, 1)
    ccy = np.bincount(lc, weights=ys, minlength=n) / np.maximum(core_px, 1)
    ids = [i for i in range(1, n) if core_px[i] * gsd * gsd >= MIN_AREA]
    if not ids:
        return pd.DataFrame()
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    rows = []
    tile = Path(path).stem
    for i in ids:
        cx, cy = ccx[i], ccy[i]
        f = blob_features(gray, hsv, i, labels, core, cx, cy, gsd)
        x, y = T * (cx + 0.5, cy + 0.5)
        rows.append({"tile": tile, "tile_path": str(path), "x": x, "y": y, **f})
    c = pd.DataFrame(rows)
    add_context(c, bounds, tile, use_bdtopo)
    return c


def add_context(c, bounds, tile, use_bdtopo):
    c["on_building"], c["bldg_dist"], c["road_margin"] = 0, 999.0, 999.0
    if not use_bdtopo:
        return
    pad = (bounds[0] - 30, bounds[1] - 30, bounds[2] + 30, bounds[3] + 30)
    pts = shapely.points(c[["x", "y"]].to_numpy())
    bld = bdtopo.buildings(pad, tile)
    if len(bld):
        tree = shapely.STRtree(bld.geometry.values)
        _, d = tree.query_nearest(pts, return_distance=True, all_matches=False)
        c["bldg_dist"] = d
        c["on_building"] = (d <= 0.5).astype(int)
    rd = bdtopo.roads(pad, tile)
    if len(rd):
        tree = shapely.STRtree(rd.geometry.values)
        (pi, ri), d = tree.query_nearest(pts, return_distance=True, all_matches=False)
        c.loc[c.index[pi], "road_margin"] = d - rd["largeur_de_chaussee"].to_numpy()[ri] / 2


FEATURES = ["core_m2", "core_d_m", "roundness", "halo_d_m", "halo_core_ratio", "star", "bg", "peak_contrast",
            "halo_sat", "core_frac_white", "local_std", "bloom_near", "bloom_mid", "bloom_ratio", "bloom_rel",
            "on_building", "bldg_dist", "road_margin"]


def _one(args):
    return tile_candidates(*args)


def all_candidates(tiles, use_bdtopo=True, workers=8):
    with Pool(workers) as pool:
        parts = pool.map(_one, [(t, use_bdtopo) for t in tiles])
    parts = [p for p in parts if len(p)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["x", "y"] + FEATURES)


def decide(c, model_dir):
    """Probability (classifier) or 0/1 (rule) that each candidate would be reported, and which was used."""
    f = Path(model_dir) / "hotspot_gbm.joblib"
    if f.exists() and len(c):
        m = joblib.load(f)
        meta = json.loads((Path(model_dir) / "hotspot_gbm.json").read_text())
        p = m.predict_proba(c[FEATURES])[:, 1]
        return p, p >= meta["threshold"], "classifier"
    r = rule(c) if len(c) else np.zeros(0, bool)
    return r.astype(float), r, "rule"


def find_hotspots(tiles, model_dir, use_bdtopo=True, workers=8):
    """detect.py entry: all candidates with prob/reportable columns, and the method used."""
    c = all_candidates(tiles, use_bdtopo, workers)
    c["prob"], c["reportable"], how = decide(c, model_dir)
    return c, how


def pr(y, yhat, n_true=None):
    """Precision over flagged candidates; recall over n_true reported hotspots (default: labelled candidates)."""
    tp = int((y & yhat).sum())
    return {"flagged": int(yhat.sum()), "tp": tp, "precision": round(float(tp / max(1, yhat.sum())), 2),
            "recall": round(float(tp / max(1, n_true or y.sum())), 2)}


def reported(name):
    """The campaign's reported hotspots inside its tiles."""
    pts = gpd.read_file(OUT / name / "artefacts.gpkg")
    return pts[(pts["art_arttype_id"] == HOTSPOT) & (pts["in_tiles"] == 1)]


def oof_threshold(c, y, folds=5):
    """Out-of-fold classifier probabilities (spatially grouped) and the threshold that maximises F1 on them."""
    from sklearn.model_selection import GroupKFold
    oof = np.zeros(len(c))
    for tr, te in GroupKFold(folds).split(c, y, tile_groups(c)):
        oof[te] = gbm().fit(c.iloc[tr][FEATURES], y[tr]).predict_proba(c.iloc[te][FEATURES])[:, 1]
    best = max(np.arange(0.1, 0.95, 0.05), key=lambda t: 2 * pr(y, oof >= t)["tp"] / (y.sum() + (oof >= t).sum()))
    return oof, float(best)


def run(name, use_bdtopo, workers):
    out = OUT / name
    cfg = load_campaigns()[name]
    c = all_candidates(cfg["tiles"], use_bdtopo, workers)
    hot = reported(name)
    d, _ = cKDTree(np.c_[hot.geometry.x, hot.geometry.y]).query(c[["x", "y"]].to_numpy()) if len(hot) else (
        np.full(len(c), np.inf), None)
    c["label"] = (d <= LABEL_RADIUS).astype(int)
    y = c["label"].to_numpy().astype(bool)
    found = int((np.array([np.hypot(c.x - g.x, c.y - g.y).min() for g in hot.geometry]) <= LABEL_RADIUS).sum())
    lines = [f"# Hotspots: {name}\n", f"{len(c)} glint candidates in {len(cfg['tiles'])} tiles; "
             f"{len(hot)} reported hotspots, {found} of them matched by a candidate (≤ {LABEL_RADIUS} m).\n"]
    c["rule"] = rule(c)
    lines.append(f"\nRule (core ≥ 2 m² and bloom ≥ 0.4): {pr(y, c['rule'].to_numpy(), len(hot))}\n")

    if y.sum() >= MIN_POSITIVES:
        oof, best = oof_threshold(c, y)
        lines.append(f"\nClassifier, grouped CV, threshold {best:.2f}: {pr(y, oof >= best, len(hot))}\n")
        joblib.dump(gbm().fit(c[FEATURES], y), out / "hotspot_gbm.joblib")
        (out / "hotspot_gbm.json").write_text(json.dumps({"threshold": float(best), "positives": int(y.sum())}))
        c["oof"] = oof
    else:
        lines.append(f"\nClassifier not trained: {int(y.sum())} labelled hotspots < {MIN_POSITIVES}; "
                     "detect.py uses the rule.\n")
    c.to_parquet(out / "hotspot_candidates.parquet")
    (out / "hotspot_report.md").write_text("".join(lines))
    print("".join(lines))


def run_transfer(train, test, min_positives):
    tag = "+".join(train)
    mdir = OUT / f"model_{tag}"
    mdir.mkdir(parents=True, exist_ok=True)
    parts = []
    for n in train:
        d = pd.read_parquet(OUT / n / "hotspot_candidates.parquet")
        d["tile"] = n + "/" + d["tile"]  # tile ids repeat across campaigns; groups must not merge them
        parts.append(d)
    c = pd.concat(parts, ignore_index=True)
    y = c["label"].to_numpy().astype(bool)
    t = pd.read_parquet(OUT / test / "hotspot_candidates.parquet")
    yt, n_true = t["label"].to_numpy().astype(bool), len(reported(test))
    lines = [f"# Hotspots: trained on {tag}, evaluated on {test}\n\n",
             f"Training: {len(c)} candidates, {int(y.sum())} labelled hotspots. Test: {len(t)} candidates, "
             f"{n_true} reported hotspots ({int(yt.sum())} matched by a candidate).\n",
             f"\nRule (core ≥ 2 m² and bloom ≥ 0.4, chosen on Eure-et-Loir): {pr(yt, rule(t), n_true)}\n"]
    for f in ("hotspot_gbm.joblib", "hotspot_gbm.json"):
        (mdir / f).unlink(missing_ok=True)  # no stale classifier: detect.py falls back to the rule
    if y.sum() >= min_positives:
        oof, best = oof_threshold(c, y)
        m = gbm().fit(c[FEATURES], y)
        p = m.predict_proba(t[FEATURES])[:, 1]
        lines.append(f"\nClassifier, threshold {best:.2f} chosen on {tag} (grouped CV there: "
                     f"{pr(y, oof >= best)}): {pr(yt, p >= best, n_true)}\n")
        joblib.dump(m, mdir / "hotspot_gbm.joblib")
        (mdir / "hotspot_gbm.json").write_text(json.dumps({"threshold": best, "positives": int(y.sum()),
                                                           "trained_on": train}))
        t["transfer_prob"] = p
        t.to_parquet(OUT / test / f"hotspot_transfer_from_{tag}.parquet")
    else:
        lines.append(f"\nClassifier not trained: {int(y.sum())} labelled hotspots in {tag} < {min_positives} "
                     "(--min-positives); detect.py with this model folder uses the rule.\n")
    (OUT / test / f"hotspot_transfer_from_{tag}.md").write_text("".join(lines))
    print("".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--campaign")
    ap.add_argument("--train", nargs="+", help="transfer mode: campaigns to train on")
    ap.add_argument("--test", help="transfer mode: campaign to evaluate on")
    ap.add_argument("--min-positives", type=int, default=MIN_POSITIVES,
                    help="transfer mode: labelled hotspots needed in the training campaigns")
    ap.add_argument("--no-bdtopo", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if args.train or args.test:
        if not (args.train and args.test):
            raise SystemExit("transfer mode needs both --train and --test")
        run_transfer(args.train, args.test, args.min_positives)
        return
    for name in load_campaigns():
        if args.campaign in (None, name):
            run(name, not args.no_bdtopo, args.workers)


if __name__ == "__main__":
    main()
