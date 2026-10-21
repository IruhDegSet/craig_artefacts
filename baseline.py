"""Phase 2 baselines, evaluated with spatially grouped cross-validation.

    python baseline.py [--campaign NAME] [--folds 5]

Scores compared (all out-of-fold):
  random      shuffled scores (mean of 20 draws), the floor
  rule        BD TOPO prior only: predicted building cut in px, + on-road, + inter-strip
  gbm_context gradient boosting on the context columns (BD TOPO, strips, parallax)
  gbm_image   gradient boosting on the cross-seam image features
  gbm_all     both

Tiles closer than GROUP_DIST m are kept in the same fold so overlapping tiles never leak.
Writes out/<campaign>/baseline_scores.parquet, baseline_report.md, gbm_all.joblib (gbm_all fitted
on every labelled station, used by infer.py and detect.py) and gbm_lines.joblib (the same without
the photo-dependent context, for seamlines given as plain polylines).
"""

import argparse

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from config import OUT, load_campaigns
from metrics import evaluate, targets

GROUP_DIST = 1500.0
CONTEXT = ["bldg_cross", "bldg_h", "bldg_dist", "road_margin", "road_importance", "on_road",
           "inter_strip", "baseline_m", "parallax_per_m", "parallax_px"]
# context that needs photo polygons with photo names; unknown when the seamlines are plain polylines
PHOTO = ["inter_strip", "baseline_m", "parallax_per_m", "parallax_px"]
CONTEXT_LINES = [c for c in CONTEXT if c not in PHOTO]


def tile_groups(st):
    c = st.groupby("tile")[["x", "y"]].mean()
    pairs = cKDTree(c.to_numpy()).query_pairs(GROUP_DIST, output_type="ndarray")
    adj = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(c), len(c)))
    _, comp = connected_components(adj, directed=False)
    return st["tile"].map(dict(zip(c.index, comp))).to_numpy()


def rule_score(st):
    return st["parallax_px"] * st["bldg_cross"] + 4 * st["on_road"] + 2 * st["inter_strip"]


def gbm():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=20,
                                          l2_regularization=1.0, class_weight="balanced", random_state=0)


def oof_gbm(X, y, groups, folds):
    score = np.full(len(X), np.nan)
    for tr, te in GroupKFold(folds).split(X, y, groups):
        m = tr[y[tr] >= 0]
        clf = gbm().fit(X.iloc[m], y[m])
        score[te] = clf.predict_proba(X.iloc[te])[:, 1]
    return score


def run(name, folds):
    out = OUT / name
    st = pd.read_parquet(out / "stations.parquet")
    feats = pd.read_parquet(out / "features.parquet")
    pts = targets(out / "artefacts.gpkg")
    y = st["label"].to_numpy()
    groups = tile_groups(st)
    print(f"{name}: {len(st)} stations, {(y == 1).sum()} positive, {len(np.unique(groups))} spatial groups")

    scores = {
        "rule": rule_score(st).to_numpy(float),
        "gbm_context": oof_gbm(st[CONTEXT], y, groups, folds),
        "gbm_image": oof_gbm(feats, y, groups, folds),
        "gbm_all": oof_gbm(pd.concat([st[CONTEXT], feats], axis=1), y, groups, folds),
        "gbm_all, no photo info": oof_gbm(pd.concat([st[CONTEXT_LINES], feats], axis=1), y, groups, folds),
    }
    rng = np.random.default_rng(0)
    rows = {}
    for on_seam in (False, True):
        tag = " (on-seam)" if on_seam else ""
        rand = [evaluate(st, rng.random(len(st)), pts, on_seam) for _ in range(20)]
        rows["random" + tag] = {k: (np.mean([r[k] for r in rand]) if isinstance(rand[0][k], float) else "")
                                for k in rand[0]}
        for k, s in scores.items():
            rows[k + tag] = evaluate(st, s, pts, on_seam)
    table = pd.DataFrame(rows).T
    num = [c for c in table if c.startswith("R@") and "/km" in c and " " not in c] + ["AP"]
    table[num] = table[num].astype(float).round(2)
    print(table.drop(columns=["km"]).to_string())

    pd.DataFrame(scores, index=st.index).to_parquet(out / "baseline_scores.parquet")
    X = pd.concat([st[CONTEXT], feats], axis=1)
    joblib.dump(gbm().fit(X[y >= 0], y[y >= 0]), out / "gbm_all.joblib")
    X = pd.concat([st[CONTEXT_LINES], feats], axis=1)
    joblib.dump(gbm().fit(X[y >= 0], y[y >= 0]), out / "gbm_lines.joblib")
    # rewrite the baselines, keep the sections train.py appended (## SeamNet ...)
    report = out / "baseline_report.md"
    old = report.read_text() if report.exists() else ""
    kept = old[old.find("\n## "):] if "\n## " in old else ""
    report.write_text(
        f"# Baselines: {name}\n\n{len(st)} stations ({st.shape[0] * 4 / 1000:.0f} km of seam), "
        f"{(y == 1).sum()} positive stations, {len(np.unique(groups))} spatial groups, {folds}-fold "
        f"grouped CV.\n\n{table.drop(columns=['km']).to_markdown()}\n{kept}")
    return table


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--campaign")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    for name in load_campaigns():
        if args.campaign in (None, name):
            run(name, args.folds)


if __name__ == "__main__":
    main()
