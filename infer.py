"""Phase 4: score a campaign's seams and export a review layer for QGIS.

    python infer.py --campaign NAME [--models-from CAMPAIGN] [--budget 10] [--no-cnn]

Needs seam_graph.py and features.py run on NAME. Models (gbm_all.joblib, seamnet.pt) are taken from
out/<models-from>/ (default: NAME itself, which is only meaningful as a demo, the scores are then
in-sample). Station scores are rank-averaged over the available models, reduced to detections by
NMS, and the best `budget` detections per km of seam are written to out/<NAME>/detections.gpkg in
the cv_artefact schema (art_ope_id = "IA", score/rank/type in art_commentaire and extra columns),
so they open in the existing contrôle visuel QGIS project.
"""

import argparse
import datetime

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

import synth
from baseline import CONTEXT
from config import CRS, OUT, STRIDE, load_campaigns
from metrics import detections
from train import SeamNet, predict

# type head class -> art_arttype_id (raccord radiométrique's id is not in the 2025 export)
# error type -> art_arttype_id (raccord radiométrique's id is not in the 2025 export)
ARTTYPE = {"batiment_coupe": 2, "vehicule_coupe": 8, "decalage": 13, "radiometrie": None,
           "hotspot": 10, "ombre_de_nuage": 14}


def cnn_scores(name, model_dir, workers):
    model = SeamNet(pretrained=False)
    model.load_state_dict(torch.load(model_dir / "seamnet.pt", map_location="cpu", weights_only=True))
    n = len(pd.read_parquet(OUT / name / "stations.parquet"))
    return predict(model, name, np.arange(n), "cpu", workers)


def heuristic_type(st):
    """Without the CNN type head: building crossed -> bâtiment coupé, road -> véhicule coupé."""
    t = np.full(len(st), 2)
    t[st["on_road"].to_numpy() == 1] = 1
    t[st["bldg_cross"].to_numpy() == 1] = 0
    return np.eye(len(synth.TYPES))[t]


def to_cv_artefact(d, campaign):
    """Detections (columns x, y, rank, score, type, ...) as a GeoDataFrame in the cv_artefact schema,
    with the scoring columns kept after the schema's own."""
    d = d.copy()
    d["art_arttype_id"] = [ARTTYPE[t] for t in d["type"]]
    d["art_campagne"] = campaign
    d["art_ortho_version"] = None
    d["art_date"] = datetime.date.today()
    d["art_renvoi"] = None
    d["art_commentaire"] = [f"auto rang {r} score {s:.2f} {t}" for r, s, t in zip(d["rank"], d["score"], d["type"])]
    d["art_correction"] = None
    d["art_correction_qualite"] = None
    d["art_ope_id"] = "IA"
    cols = ["art_campagne", "art_ortho_version", "art_date", "art_arttype_id", "art_renvoi", "art_commentaire",
            "art_correction", "art_correction_qualite", "art_ope_id", "rank", "score", "type", "tile",
            "img_L", "img_R"] + [c for c in ("p_gbm", "p_cnn") if c in d]
    for c in ("img_L", "img_R"):
        if c not in d:
            d[c] = None
    return gpd.GeoDataFrame(d[cols], geometry=gpd.points_from_xy(d.x, d.y), crs=CRS)


def run(name, models_from, budget, use_cnn, workers):
    out, mdir = OUT / name, OUT / models_from
    st = pd.read_parquet(out / "stations.parquet")
    feats = pd.read_parquet(out / "features.parquet")
    gbm = joblib.load(mdir / "gbm_all.joblib")
    s_gbm = gbm.predict_proba(pd.concat([st[CONTEXT], feats], axis=1)[gbm.feature_names_in_])[:, 1]
    scores, ptype = [rankdata(s_gbm) / len(st)], heuristic_type(st)
    st["p_gbm"] = s_gbm
    if use_cnn and (mdir / "seamnet.pt").exists():
        s_cnn, ptype = cnn_scores(name, mdir, workers)
        scores.append(rankdata(s_cnn) / len(st))
        st["p_cnn"] = s_cnn
    st["score"] = np.mean(scores, axis=0)
    st[["x", "y", "score"] + [c for c in ("p_gbm", "p_cnn") if c in st]].to_parquet(out / "scored.parquet")

    km = len(st) * STRIDE / 1000
    det = detections(st[["x", "y"]].to_numpy(), st["score"].to_numpy())[: max(1, round(budget * km))]
    d = st.iloc[det].copy()
    d["rank"] = np.arange(1, len(d) + 1)
    d["type"] = [synth.TYPES[i] for i in ptype[det].argmax(1)]
    gdf = to_cv_artefact(d, name)
    gdf.to_file(out / "detections.gpkg", layer="cv_artefact")
    print(f"{name}: {len(st)} stations (~{km:.0f} km), {len(gdf)} detections at {budget}/km "
          f"-> {out / 'detections.gpkg'}; types {gdf['type'].value_counts().to_dict()}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--models-from", help="campaign whose out/ folder holds the models (default: --campaign)")
    ap.add_argument("--budget", type=float, default=10, help="detections per km of seam")
    ap.add_argument("--no-cnn", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    assert args.campaign in load_campaigns(), f"{args.campaign} not in campaigns.yaml"
    run(args.campaign, args.models_from or args.campaign, args.budget, not args.no_cnn, args.workers)


if __name__ == "__main__":
    main()
