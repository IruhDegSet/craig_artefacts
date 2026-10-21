"""Detect orthophoto errors on tiles: seam errors, hotspots and cloud shadows. The one command for new data.

    python detect.py --tiles "/data/tiles/*.tif" --seamlines seamlines.shp --out /data/results
                     [--checks seams,hotspots,clouds] [--threshold 0.7] [--budget 10] [--layer NAME]
                     [--image-field Image] [--models out/eure_et_loir_nord_2025] [--no-bdtopo] [--no-cnn]

Inputs
  tiles       GeoTIFFs in EPSG:2154, RGB or RGBA, ~5 cm (e.g. 4000 x 4000 px = 200 m); globs or paths
  seamlines   .shp or .gpkg in any CRS (needed by the seams and clouds checks), either
                - polygons, one per source photo (mosaic graph): best, photo names in --image-field
                  give the strip and parallax context;
                - polylines: the seams themselves; photo context is then unknown and a GBM trained
                  without it is used (grouped CV: R@5/km 0.66 instead of 0.69 on-seam).

Checks (all by default)
  seams     bâtiment coupé, véhicule coupé, décalage/mosaïquage: SeamNet probability >= --threshold
            at seam stations, after 8 m non-maximum suppression
  hotspots  specular glints that an operator would report: saturated core >= 2 m2 that glows beyond
            its outline (hotspot.py; a classifier replaces this rule once trained)
  clouds    EXPERIMENTAL cloud shadows cut by a seam: >= 80 m of seam where the same open land is
            darker on one side (cloud.py)

Outputs in --out
  errors.gpkg   the automatic result: one point per error, cv_artefact schema (art_arttype_id 2/8/13
                seams, 10 hotspot, 14 cloud shadow) + type, score, tile
  review.gpkg   for an operator: cv_artefact (ranked seam candidates, --budget per km of seam),
                hotspot_candidates (every glint with its features), ombre_de_nuage_stretches and
                ombre_de_nuage_patches (all shadow candidates)
  scored.gpkg   every seam station with its probabilities
  summary.csv   per tile: seam length and errors by type

The expected precision/recall of the seam check at the chosen threshold is printed from the model
folder's thresholds.json (out-of-fold scores of the campaign the model was trained on).
"""

import argparse
import glob
import json
import subprocess
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import rasterio
from scipy.stats import rankdata
from shapely.ops import nearest_points

import features
import synth
from baseline import CONTEXT, CONTEXT_LINES, PHOTO
from config import CRS, OUT, STRIDE
from infer import cnn_scores, heuristic_type, to_cv_artefact
from metrics import detections
from seam_graph import dedupe, tile_stations
from cloud import STRETCH_M, find_shadows, stretches
from hotspot import FEATURES, find_hotspots

NEUTRAL_CONTEXT = {"bldg_cross": 0, "bldg_h": 0.0, "bldg_dist": 999.0, "road_margin": 999.0,
                   "road_importance": 9, "on_road": 0}


def expand(patterns):
    paths = []
    for p in patterns:
        paths += sorted(glob.glob(p)) if any(ch in p for ch in "*?[") else [p]
    return [str(Path(p).resolve()) for p in dict.fromkeys(paths)]


def check_tiles(tiles):
    if not tiles:
        raise SystemExit("no tile matches --tiles")
    for t in tiles:
        with rasterio.open(t) as ds:
            if ds.crs is None or ds.crs.to_epsg() != 2154:
                raise SystemExit(f"{t}: CRS is {ds.crs}, EPSG:2154 (Lambert-93) is required; "
                                 "reproject with gdalwarp -t_srs EPSG:2154")
            if ds.count < 3:
                raise SystemExit(f"{t}: {ds.count} band(s), RGB or RGBA is required")


def build_stations(tiles, cfg, out, use_bdtopo):
    parts, edges = [], []
    for i, t in enumerate(tiles, 1):
        st, e = tile_stations(cfg, t, use_bdtopo)
        if st.empty:
            print(f"  [{i}/{len(tiles)}] {Path(t).name}: no seam crosses this tile")
            continue
        parts.append(st)
        edges += e
        print(f"  [{i}/{len(tiles)}] {Path(t).name}: {len(st)} stations")
    if not parts:
        raise SystemExit("no seam crosses any tile: check that the seamlines cover the tiles (and --layer)")
    st = dedupe(pd.concat(parts, ignore_index=True))
    if not use_bdtopo:
        for c, v in NEUTRAL_CONTEXT.items():
            st[c] = v
        st["parallax_px"] = 0.0 * st["parallax_per_m"]  # NaN stays NaN in lines mode
    st["campaign"] = out.name
    st.to_parquet(out / "stations.parquet")
    gpd.GeoDataFrame(edges, geometry="geometry", crs=CRS).to_file(out / "edges.gpkg")
    (out / "tiles.txt").write_text("\n".join(tiles) + "\n")
    subprocess.run(["gdalbuildvrt", "-q", "-resolution", "highest", "-input_file_list", str(out / "tiles.txt"),
                    str(out / "mosaic.vrt")], check=True)
    return st


def expected(models, threshold):
    f = models / "thresholds.json"
    if not f.exists():
        return ""
    t = json.loads(f.read_text())
    rows = [r for r in t["rows"] if r["threshold"] <= threshold + 1e-9]
    if not rows:
        return ""
    r = rows[-1]
    return (f"expected at threshold {r['threshold']} on {t['campaign']} ({t['source']}): ~{r['per_km']} errors/km, "
            f"precision >= {r['precision']:.0%}, recall {r['recall']:.0%}")


def seam_errors(args, tiles, st, out, models):
    """Score the seam stations; returns (errors, review candidates, info line)."""
    print("  cross-seam features")
    features.build(out, args.workers)  # OUT / <absolute path> is that path
    feats = pd.read_parquet(out / "features.parquet")
    photo = not st[PHOTO].isna().all().any()
    gbm = joblib.load(models / ("gbm_all.joblib" if photo else "gbm_lines.joblib"))
    X = pd.concat([st[CONTEXT if photo else CONTEXT_LINES], feats], axis=1)
    st["p_gbm"] = gbm.predict_proba(X[gbm.feature_names_in_])[:, 1]
    use_cnn = not args.no_cnn and (models / "seamnet.pt").exists()
    print(f"  scoring ({'SeamNet + GBM' if use_cnn else 'GBM'})")
    if use_cnn:
        st["p_cnn"], ptype = cnn_scores(out, models, args.workers)
        st["prob"] = st["p_cnn"]
        st["score"] = (rankdata(st["p_cnn"]) + rankdata(st["p_gbm"])) / (2 * len(st))
    else:
        ptype = heuristic_type(st)
        st["prob"] = st["score"] = st["p_gbm"]
    st["type"] = [synth.TYPES[i] for i in ptype.argmax(1)]
    xy = st[["x", "y"]].to_numpy()

    keep = ["x", "y", "tile", "img_L", "img_R", "type", "prob", "score", "p_gbm"] + (["p_cnn"] if use_cnn else [])
    gpd.GeoDataFrame(st[keep], geometry=gpd.points_from_xy(st.x, st.y), crs=CRS).to_file(out / "scored.gpkg")
    det = detections(xy, st["prob"].to_numpy())
    err = st.iloc[det[st["prob"].to_numpy()[det] >= args.threshold]].copy()
    err["score"] = err["prob"]
    km = len(st) * STRIDE / 1000
    rev = st.iloc[detections(xy, st["score"].to_numpy())[: max(1, round(args.budget * km))]].copy()
    rev["rank"] = np.arange(1, len(rev) + 1)
    info = (f"seams: {len(err)} errors on {km:.1f} km ({'polygons with photo names' if photo else 'no photo information'}"
            f", {'SeamNet' if use_cnn else 'GBM'} >= {args.threshold})")
    if use_cnn:
        info += "\n    " + expected(models, args.threshold)
    return err, rev, info


def hotspot_errors(args, tiles, out, models):
    c, how = find_hotspots(tiles, models, not args.no_bdtopo, args.workers)
    keep = ["x", "y", "tile", "prob", "reportable"] + FEATURES
    gpd.GeoDataFrame(c[keep], geometry=gpd.points_from_xy(c.x, c.y), crs=CRS).to_file(
        out / "review.gpkg", layer="hotspot_candidates")
    err = c[c["reportable"]].copy()
    err["type"], err["score"] = "hotspot", err["prob"]
    return err, f"hotspots: {len(err)} reportable of {len(c)} glints ({how})"


def cloud_errors(args, tiles, out):
    edges = gpd.read_file(out / "edges.gpkg").geometry
    cands = find_shadows(tiles, edges, not args.no_bdtopo)
    st = stretches(cands)
    if len(cands):
        cands[cands["kind"] == "dark_patch"].to_file(out / "review.gpkg", layer="ombre_de_nuage_patches")
    if len(st):
        st.to_file(out / "review.gpkg", layer="ombre_de_nuage_stretches")
    sh = st[st["shadow"]]
    # a stretch is a MultiLineString: report the point of it nearest its centroid
    pts = gpd.GeoSeries([nearest_points(g, g.centroid)[0] for g in sh.geometry], crs=CRS)
    err = pd.DataFrame({"x": pts.x, "y": pts.y, "tile": sh["tile"].to_numpy(), "score": sh["score"].to_numpy(),
                        "type": "ombre_de_nuage"})
    return err, f"cloud shadows (experimental): {len(err)} seam stretches >= {STRETCH_M:.0f} m"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tiles", nargs="+", required=True, help="tile paths or globs (quote the globs)")
    ap.add_argument("--seamlines", help="seamline polygons or polylines, .shp or .gpkg (seams and clouds checks)")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--checks", default="seams,hotspots,clouds", help="comma-separated: seams, hotspots, clouds")
    ap.add_argument("--layer", help="layer of a multi-layer gpkg")
    ap.add_argument("--image-field", default="Image", help="photo name field of seamline polygons")
    ap.add_argument("--models", default=str(OUT / "eure_et_loir_nord_2025"),
                    help="folder with seamnet.pt, gbm_*.joblib, thresholds.json (and hotspot_gbm.* once trained)")
    ap.add_argument("--threshold", type=float, default=0.7, help="SeamNet probability to flag a seam error")
    ap.add_argument("--budget", type=float, default=10, help="review.gpkg: seam candidates per km of seam")
    ap.add_argument("--no-bdtopo", action="store_true", help="skip BD TOPO context (offline)")
    ap.add_argument("--no-cnn", action="store_true", help="GBM only (its probability is then thresholded)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    checks = {c.strip() for c in args.checks.split(",") if c.strip()}
    if checks - {"seams", "hotspots", "clouds"}:
        raise SystemExit(f"unknown check(s): {sorted(checks - {'seams', 'hotspots', 'clouds'})}")
    if checks & {"seams", "clouds"} and not args.seamlines:
        raise SystemExit("--seamlines is required for the seams and clouds checks (or use --checks hotspots)")
    tiles = expand(args.tiles)
    check_tiles(tiles)
    out, models = Path(args.out).resolve(), Path(args.models).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "review.gpkg").unlink(missing_ok=True)

    errors, infos, st = [], [], None
    if checks & {"seams", "clouds"}:
        print(f"seam stations on {len(tiles)} tile(s)")
        cfg = {"seamlines": str(Path(args.seamlines).resolve()), "layer": args.layer,
               "image_field": args.image_field}
        st = build_stations(tiles, cfg, out, not args.no_bdtopo)
    if "seams" in checks:
        print("seam errors")
        err, rev, info = seam_errors(args, tiles, st, out, models)
        errors.append(err)
        infos.append(info)
        to_cv_artefact(rev, out.name).to_file(out / "review.gpkg", layer="cv_artefact")
    if "hotspots" in checks:
        print("hotspots")
        err, info = hotspot_errors(args, tiles, out, models)
        errors.append(err)
        infos.append(info)
    if "clouds" in checks:
        print("cloud shadows")
        err, info = cloud_errors(args, tiles, out)
        errors.append(err)
        infos.append(info)

    err = pd.concat([e for e in errors if len(e)], ignore_index=True) if any(len(e) for e in errors) else \
        pd.DataFrame(columns=["x", "y", "tile", "type", "score"])
    err = err.sort_values("score", ascending=False, kind="stable").reset_index(drop=True)
    err["rank"] = np.arange(1, len(err) + 1)
    to_cv_artefact(err, out.name).to_file(out / "errors.gpkg", layer="cv_artefact")

    summary = pd.DataFrame(index=pd.Index(sorted({Path(t).stem for t in tiles}), name="tile"))
    if st is not None:
        summary["seam_m"] = st.groupby("tile").size().reindex(summary.index).fillna(0) * STRIDE
    counts = err.groupby(["tile", "type"]).size().unstack(fill_value=0)
    summary = summary.join(counts).fillna(0)
    summary["errors"] = summary[list(counts.columns)].sum(axis=1) if len(counts.columns) else 0
    summary = summary.astype({c: int for c in summary.columns if c != "seam_m"})
    summary.to_csv(out / "summary.csv")

    print(f"\n{len(err)} errors on {len(tiles)} tile(s): {err['type'].value_counts().to_dict()}")
    for i in infos:
        print("  " + i)
    also = [f for f in ("review.gpkg", "scored.gpkg", "summary.csv") if (out / f).exists()]
    print(f"-> {out / 'errors.gpkg'}  (also {', '.join(also)})")


if __name__ == "__main__":
    main()
