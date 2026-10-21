"""Seam graph: stations every STRIDE m along the seamlines inside each tile, with context and labels.

    python seam_graph.py [--campaign NAME] [--no-bdtopo]

For each campaign in campaigns.yaml, writes under out/<campaign>/
    stations.parquet / stations.gpkg  one row per station (position, tangent, photos on each side,
                                      BD TOPO context, distance to artefacts, label 1/0/-1)
    edges.gpkg                        seam lines clipped to the tiles
    artefacts.gpkg                    the campaign's points, with merged instance ids and seam distance
    mosaic.vrt                        all tiles of the campaign, read by crops.py

A seam is the shared boundary of two neighbouring photo polygons: the boundaries are noded and
merged, and the photo on each side is found by a point-in-polygon test SIDE_OFFSET m off the line.
Left is the +normal side when walking along the line (normal = tangent rotated 90° anticlockwise).
"""

import argparse
import subprocess

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import rasterio
import shapely
from rasterio.warp import transform_bounds
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from shapely.geometry import box
from shapely.ops import linemerge, substring, unary_union

import bdtopo
from config import (CRS, DEDUPE, FLYING_HEIGHT, GSD, IGNORE_RADIUS, MERGE_RADIUS, OTHER_IGNORE, OUT,
                    POS_RADIUS, SEAM_TYPES, SIDE_OFFSET, STRIDE, TANGENT_HALF, load_campaigns)


def read_seams(cfg, bounds):
    """Seamline features intersecting bounds (EPSG:2154), reprojected to EPSG:2154.

    cfg["seamlines"] is one file or a list of files, each in its own CRS; they are read together.
    Photo polygons (the mosaic graph) or plain polylines. With polygons, `image` is the photo name and
    `strip` its flight strip when the image field exists (else image = row id, strip = None); with
    lines both are None. The result carries attrs["kind"] = "polygons" or "lines"."""
    layer = cfg.get("layer")
    paths = cfg["seamlines"] if isinstance(cfg["seamlines"], list) else [cfg["seamlines"]]
    parts = []
    for path in paths:
        info = pyogrio.read_info(path, layer=layer)
        bb = transform_bounds(CRS, info["crs"], *bounds)
        parts.append(gpd.read_file(path, bbox=bb, layer=layer).to_crs(CRS))
    g = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
    g = g[g.geometry.notna() & ~g.geometry.is_empty].reset_index(drop=True)
    lines = g.geom_type.str.contains("LineString").all() if len(g) else "LineString" in info["geometry_type"]
    field = cfg.get("image_field", "Image")
    if lines:
        g["image"], g["strip"] = None, None
    elif field in g:
        g["image"] = g[field].astype(str)
        g["strip"] = g["image"].str.split("_").str[0]
    else:
        g["image"], g["strip"] = g.index.astype(str), None
    g = g[["image", "strip", "geometry"]]
    g["cx"], g["cy"] = g.geometry.centroid.x, g.geometry.centroid.y
    g.attrs["kind"] = "lines" if lines else "polygons"
    return g


def seam_lines(seams, tile_box):
    """Merged seam lines inside the tile: shared polygon boundaries, or the polylines themselves."""
    geoms = list(seams.geometry) if seams.attrs.get("kind") == "lines" else [p.boundary for p in seams.geometry]
    if not geoms:
        return []
    lines = unary_union(geoms).intersection(tile_box)
    if lines.is_empty:
        return []
    if lines.geom_type == "MultiLineString":
        lines = linemerge(lines)
    parts = getattr(lines, "geoms", [lines])
    return [l for l in parts if l.geom_type == "LineString" and l.length > 0.5]


def stations_on(line):
    """Arc-lengths, points and unit tangents every STRIDE m along a line."""
    L = line.length
    s = np.arange(STRIDE / 2, L, STRIDE) if L > STRIDE / 2 else np.array([L / 2])
    p = shapely.get_coordinates(shapely.line_interpolate_point(line, s))
    a = shapely.get_coordinates(shapely.line_interpolate_point(line, np.clip(s - TANGENT_HALF, 0, L)))
    b = shapely.get_coordinates(shapely.line_interpolate_point(line, np.clip(s + TANGENT_HALF, 0, L)))
    t = b - a
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
    return s, p, t


def side_index(polys, xy):
    """Index of the polygon containing each point, -1 if none."""
    tree = shapely.STRtree(polys.geometry.values)
    pi, gi = tree.query(shapely.points(xy), predicate="within")
    out = np.full(len(xy), -1)
    out[pi] = gi
    return out


def tile_stations(cfg, tile_path, use_bdtopo=True):
    tile = tile_path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    with rasterio.open(tile_path) as ds:
        bounds = tuple(ds.bounds)
    tile_box = box(*bounds)
    polys = read_seams(cfg, bounds)
    is_lines = polys.attrs["kind"] == "lines"
    rows, edges = [], []
    for k, line in enumerate(seam_lines(polys, tile_box)):
        s, p, t = stations_on(line)
        if is_lines:  # no photo polygons: every station is kept, photo information is unknown
            keep, li, ri = np.ones(len(s), bool), None, None
        else:
            n = np.stack([-t[:, 1], t[:, 0]], axis=1)
            li = side_index(polys, p + SIDE_OFFSET * n)
            ri = side_index(polys, p - SIDE_OFFSET * n)
            keep = (li >= 0) & (ri >= 0) & (li != ri)
        if not keep.any():
            continue
        edge = f"{tile}_{k}"
        edges.append({"edge": edge, "tile": tile, "geometry": line})
        for j in np.flatnonzero(keep):
            row = {"tile": tile, "tile_path": tile_path, "edge": edge, "s": s[j],
                   "x": p[j, 0], "y": p[j, 1], "angle": float(np.arctan2(t[j, 1], t[j, 0])),
                   "img_L": None, "img_R": None, "strip_L": None, "strip_R": None,
                   "inter_strip": np.nan, "baseline_m": np.nan,
                   "_seg": substring(line, max(s[j] - 2, 0), min(s[j] + 2, line.length))}
            if not is_lines:
                L, R = polys.iloc[li[j]], polys.iloc[ri[j]]
                row.update({"img_L": L.image, "img_R": R.image, "strip_L": L.strip, "strip_R": R.strip,
                            "baseline_m": float(np.hypot(L.cx - R.cx, L.cy - R.cy))})
                if L.strip is not None:
                    row["inter_strip"] = int(L.strip != R.strip)
            rows.append(row)
    st = pd.DataFrame(rows)
    if st.empty:
        return st, edges
    st["parallax_per_m"] = st["baseline_m"] / FLYING_HEIGHT / GSD  # px of cut per m of object height
    if use_bdtopo:
        add_context(st, bounds, tile)
    return st.drop(columns="_seg"), edges


def add_context(st, bounds, tile):
    """Building crossed by the seam (and its height), distance to buildings and roads."""
    pad = (bounds[0] - 30, bounds[1] - 30, bounds[2] + 30, bounds[3] + 30)
    bld = bdtopo.buildings(pad, tile)
    rd = bdtopo.roads(pad, tile)
    pts = shapely.points(st[["x", "y"]].to_numpy())
    segs = st["_seg"].to_numpy()

    st["bldg_cross"], st["bldg_h"], st["bldg_dist"] = 0, 0.0, 999.0
    if len(bld):
        tree = shapely.STRtree(bld.geometry.values)
        si, bi = tree.query(segs, predicate="intersects")
        h = bld["hauteur"].to_numpy()
        hmax = pd.Series(h[bi]).groupby(si).max()
        st.loc[st.index[hmax.index], "bldg_cross"] = 1
        st.loc[st.index[hmax.index], "bldg_h"] = hmax.to_numpy()
        _, d = tree.query_nearest(pts, return_distance=True, all_matches=False)
        st["bldg_dist"] = d
    st["road_margin"], st["road_importance"] = 999.0, 9
    if len(rd):
        tree = shapely.STRtree(rd.geometry.values)
        (pi, ri), d = tree.query_nearest(pts, return_distance=True, all_matches=False)
        half = rd["largeur_de_chaussee"].to_numpy()[ri] / 2
        st.loc[st.index[pi], "road_margin"] = d - half
        imp = pd.to_numeric(rd["importance"], errors="coerce").fillna(9).to_numpy()
        st.loc[st.index[pi], "road_importance"] = imp[ri]
    st["on_road"] = (st["road_margin"] <= 1.0).astype(int)
    st["parallax_px"] = st["bldg_h"] * st["parallax_per_m"] * st["bldg_cross"]


def artefact_instances(cfg):
    """The campaign's points; same-type points within MERGE_RADIUS share an instance id."""
    pts = gpd.read_file(cfg["artefacts"], where=f"art_campagne = '{cfg['campaign']}'", fid_as_index=True)
    pts = pts.to_crs(CRS).reset_index().rename(columns={"index": "art_id", "fid": "art_id"})
    xy = np.c_[pts.geometry.x, pts.geometry.y]
    pairs = cKDTree(xy).query_pairs(MERGE_RADIUS, output_type="ndarray")
    same = pts["art_arttype_id"].to_numpy()
    pairs = pairs[same[pairs[:, 0]] == same[pairs[:, 1]]]
    adj = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(pts), len(pts)))
    _, comp = connected_components(adj, directed=False)
    first = pd.Series(pts["art_id"].to_numpy()).groupby(comp).transform("min").to_numpy()
    pts["instance"] = first
    pts["is_seam_type"] = pts["art_arttype_id"].isin(list(SEAM_TYPES)).astype(int)
    return pts


def label(st, pts):
    seam = pts[pts["is_seam_type"] == 1]
    other = pts[pts["is_seam_type"] == 0]
    xy = st[["x", "y"]].to_numpy()
    st["d_seam_pt"], st["seam_instance"], st["seam_type"] = np.inf, -1, -1
    if len(seam):
        d, i = cKDTree(np.c_[seam.geometry.x, seam.geometry.y]).query(xy)
        st["d_seam_pt"] = d
        st["seam_instance"] = seam["instance"].to_numpy()[i]
        st["seam_type"] = seam["art_arttype_id"].to_numpy()[i]
    st["d_other_pt"] = cKDTree(np.c_[other.geometry.x, other.geometry.y]).query(xy)[0] if len(other) else np.inf
    st["label"] = 0
    st.loc[(st["d_seam_pt"] <= IGNORE_RADIUS) | (st["d_other_pt"] <= OTHER_IGNORE), "label"] = -1
    st.loc[st["d_seam_pt"] <= POS_RADIUS, "label"] = 1
    st.loc[st["d_seam_pt"] > POS_RADIUS, ["seam_instance", "seam_type"]] = -1


def dedupe(st):
    """Overlapping tiles see the same seam: keep a station only once (first tile wins)."""
    xy = st[["x", "y"]].to_numpy()
    tiles = st["tile"].to_numpy()
    drop = np.zeros(len(st), bool)
    for i, j in cKDTree(xy).query_pairs(DEDUPE):
        a, b = min(i, j), max(i, j)
        if tiles[a] != tiles[b] and not drop[a]:
            drop[b] = True
    return st[~drop].reset_index(drop=True)


def seam_distance(pts, edges):
    if not len(edges):
        pts["d_seam_line"] = np.inf
        return
    lines = shapely.MultiLineString(list(edges.geometry))
    pts["d_seam_line"] = shapely.distance(pts.geometry.values, lines)


def build(name, cfg, use_bdtopo=True):
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    all_st, all_edges = [], []
    for i, path in enumerate(cfg["tiles"], 1):
        st, edges = tile_stations(cfg, path, use_bdtopo)
        all_st.append(st)
        all_edges += edges
        print(f"[{i}/{len(cfg['tiles'])}] {path.rsplit('/', 2)[-2:]}: {len(st)} stations, {len(edges)} edges")
    st = dedupe(pd.concat(all_st, ignore_index=True))
    st["campaign"] = name
    pts = artefact_instances(cfg)
    label(st, pts)
    edges = gpd.GeoDataFrame(all_edges, geometry="geometry", crs=CRS)
    seam_distance(pts, edges)
    tiles_union = unary_union([box(*rasterio.open(p).bounds) for p in cfg["tiles"]])
    pts["in_tiles"] = pts.geometry.within(tiles_union).astype(int)

    st.to_parquet(out / "stations.parquet")
    gpd.GeoDataFrame(st, geometry=gpd.points_from_xy(st.x, st.y), crs=CRS).to_file(out / "stations.gpkg")
    edges.to_file(out / "edges.gpkg")
    pts.to_file(out / "artefacts.gpkg")
    (out / "tiles.txt").write_text("\n".join(cfg["tiles"]) + "\n")
    subprocess.run(["gdalbuildvrt", "-q", "-resolution", "highest", "-input_file_list",
                    str(out / "tiles.txt"), str(out / "mosaic.vrt")], check=True)

    km = len(st) * STRIDE / 1000
    seam_in = pts[(pts["in_tiles"] == 1) & (pts["is_seam_type"] == 1)]
    print(f"{name}: {len(st)} stations (~{km:.0f} km of seam), labels "
          f"{st['label'].value_counts().to_dict()}, {seam_in['instance'].nunique()} seam-type instances in tiles "
          f"({(seam_in.groupby('instance')['d_seam_line'].min() <= POS_RADIUS).sum()} within {POS_RADIUS} m of a seam)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--campaign", help="only this campaign")
    ap.add_argument("--no-bdtopo", action="store_true", help="skip the BD TOPO context (offline)")
    args = ap.parse_args()
    for name, cfg in load_campaigns().items():
        if args.campaign in (None, name):
            build(name, cfg, not args.no_bdtopo)


if __name__ == "__main__":
    main()
