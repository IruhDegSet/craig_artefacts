"""Cloud shadows (ombre de nuage): EXPERIMENTAL, unsupervised, calibrated on 3 examples only.

    python cloud.py --campaign NAME            evaluate on a labelled campaign (out/<NAME>/ from seam_graph.py)

A cloud shadow belongs to one photo. It becomes a reportable artefact where the mosaic shows it,
typically cut by a seamline ("déplacer la ligne de mosaiquage"), or as a large soft dark patch.
Two cues, both on open land (away from BD TOPO buildings and roads, where gray surfaces and small
shadows confuse brightness):

  seam step   along each seam, 20 m segments: the land 3-25 m each side is darker on one side by
              >= STEP, with the same chromaticity (a shadow scales light, it does not change colour)
              and texture contrast scaled like brightness (the same field, dimmer)
  dark patch  open land >= PATCH_M2 darker by >= DEPTH than the bright part of the tile's open land,
              with soft edges (except where they run along a seam)

Decision: stepped seam segments are merged into contiguous stretches; a stretch >= STRETCH_M is a
cloud shadow (errors.gpkg, type 14). Dark patches are kept as review candidates only: alone they do
not separate shadows from darker fields (Eure-et-Loir: 1 shadow tile among 21 tiles with a patch).
"""

import argparse
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from rasterio.features import rasterize, shapes
from scipy import ndimage as ndi
from shapely.geometry import box, shape
from shapely.ops import substring

import bdtopo
from config import CRS, OUT, load_campaigns

RES = 0.25          # m: working resolution
STEP = 0.12         # minimum relative darkening across a seam
CHROM = 0.015       # max chromaticity difference between the two sides
TEXTURE = 0.4       # max |log(texture ratio) - log(brightness ratio)|
BUILT = 0.2         # max share of a band's samples within 3 m of a building or road
STRETCH_M = 80.0    # m of contiguous stepped seam to call a cloud shadow (Eure-et-Loir: shadows 111-130 m,
                    # anything else <= 51 m)
PATCH_M2 = 1500.0
DEPTH = 0.15
OMBRE = 14          # art_arttype_id


def read_tile(path):
    with rasterio.open(path) as ds:
        f = abs(ds.transform.a) / RES  # < 1: downsampling
        h, w = int(ds.height * f), int(ds.width * f)
        x = np.moveaxis(ds.read([1, 2, 3], out_shape=(3, h, w)), 0, -1).astype(np.float32)
        valid = ds.read(4, out_shape=(h, w)) > 0 if ds.count > 3 else np.ones((h, w), bool)
        T = ds.transform * ds.transform.scale(ds.width / w, ds.height / h)
        bounds = tuple(ds.bounds)
    return x, valid, T, bounds


def built_mask(shape_hw, T, bounds, tile, use_bdtopo):
    """Pixels within 3 m of a BD TOPO building or road."""
    if not use_bdtopo:
        return np.zeros(shape_hw, bool)
    pad = (bounds[0] - 30, bounds[1] - 30, bounds[2] + 30, bounds[3] + 30)
    geoms = [g.buffer(3) for g in bdtopo.buildings(pad, tile).geometry]
    rd = bdtopo.roads(pad, tile)
    geoms += [g.buffer(w / 2 + 3) for g, w in zip(rd.geometry, rd["largeur_de_chaussee"])]
    if not geoms:
        return np.zeros(shape_hw, bool)
    return rasterize(((g, 1) for g in geoms), out_shape=shape_hw, transform=T, fill=0, dtype="uint8") > 0


def seam_steps(x, valid, built, T, edges, tile):
    """One row per 20 m seam segment that passes the shadow tests."""
    g = x.mean(-1)
    S = x.sum(-1) + 1
    chrom = np.stack([x[..., 0] / S, x[..., 1] / S], -1)
    tex = cv2.blur((g - cv2.blur(g, (5, 5))) ** 2, (9, 9)) ** 0.5
    inv = ~T
    h, w = g.shape
    rows = []
    for line in edges:
        L = line.length
        for s0 in np.arange(0, max(L - 20, 1), 10):
            seg = substring(line, s0, min(L, s0 + 20))
            if seg.length < 10:
                continue
            c = seg.interpolate(0.5, normalized=True)
            a0, a1 = seg.interpolate(0), seg.interpolate(seg.length)
            t = np.array([a1.x - a0.x, a1.y - a0.y])
            t /= np.linalg.norm(t) + 1e-9
            n = np.array([-t[1], t[0]])
            sides = []
            for side in (1, -1):
                d, u = np.meshgrid(np.arange(3, 25, 2), np.arange(-8, 9, 2))
                px = c.x + side * n[0] * d + t[0] * u
                py = c.y + side * n[1] * d + t[1] * u
                cc, rr = inv * (px.ravel(), py.ravel())
                rr, cc = rr.astype(int), cc.astype(int)
                ok = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
                rr, cc = rr[ok], cc[ok]
                if len(rr) < 40 or not valid[rr, cc].all() or built[rr, cc].mean() > BUILT:
                    break
                sides.append((np.median(g[rr, cc]), np.median(chrom[rr, cc], 0), np.median(tex[rr, cc]) + 1))
            if len(sides) < 2:
                continue
            (g1, c1, t1), (g2, c2, t2) = sides
            step = 1 - min(g1, g2) / max(g1, g2)
            dchrom = float(np.abs(c1 - c2).max())
            tex_ok = abs(np.log(t1 / t2) - np.log(g1 / g2)) <= TEXTURE
            if step >= STEP and dchrom <= CHROM and tex_ok:
                rows.append({"tile": tile, "kind": "seam_step", "score": float(step), "geometry": seg})
    return rows


def dark_patches(x, valid, built, T, edges_mask, tile):
    """Large dark soft-edged regions of open land."""
    g = x.mean(-1)
    k = int(1 / RES)
    h, w = (g.shape[0] // k) * k, (g.shape[1] // k) * k
    G = g[:h, :w].reshape(h // k, k, w // k, k).mean((1, 3))
    tex = g[:h, :w].reshape(h // k, k, w // k, k).std((1, 3))
    rgb = x[:h, :w].reshape(h // k, k, w // k, k, 3).mean((1, 3))
    exg = 2 * rgb[..., 1] - rgb[..., 0] - rgb[..., 2]
    ok = (valid[:h, :w].reshape(h // k, k, w // k, k).all((1, 3))
          & ~built[:h, :w].reshape(h // k, k, w // k, k).any((1, 3))
          & (ndi.median_filter(tex, 5) < 12) & (exg < 25))
    if ok.mean() < 0.1:
        return []
    num = ndi.gaussian_filter(np.where(ok, G, 0), 6)
    den = ndi.gaussian_filter(ok.astype(float), 6)
    Ls = num / np.maximum(den, 1e-3)
    r = Ls / np.percentile(Ls[ok], 80)
    dark = ok & (r < 1 - DEPTH)
    lab, n = ndi.label(dark)
    seam = edges_mask[:h, :w].reshape(h // k, k, w // k, k).any((1, 3))
    seam = ndi.binary_dilation(seam, iterations=3)
    grad = np.hypot(*np.gradient(ndi.gaussian_filter(G, 1)))
    out = []
    Tk = T * T.scale(k, k)
    for i in range(1, n + 1):
        m = lab == i
        if m.sum() < PATCH_M2:
            continue
        ring = ndi.binary_dilation(m, iterations=3) & ~m & ~seam
        depth = float(1 - r[m].mean())
        sharp = float(grad[ring].mean() / max(depth, 1e-3) / 100) if ring.any() else 0.0
        if sharp >= 1.0:
            continue  # a sharp-edged dark area is a field or a surface, not a shadow
        poly = shapely.union_all([shape(s) for s, v in shapes(m.astype(np.uint8), mask=m, transform=Tk) if v])
        out.append({"tile": tile, "kind": "dark_patch", "score": depth, "area_m2": float(m.sum()),
                    "geometry": poly.simplify(1.0)})
    return out


def tile_shadows(path, edges, use_bdtopo=True):
    """Shadow candidates of one tile; `edges` = seam lines (EPSG:2154) intersecting it."""
    tile = Path(path).stem
    x, valid, T, bounds = read_tile(path)
    built = built_mask(valid.shape, T, bounds, tile, use_bdtopo)
    lines = [e for e in edges if e.intersects(box(*bounds))]
    emask = (rasterize(((l, 1) for l in lines), out_shape=valid.shape, transform=T, fill=0, dtype="uint8") > 0
             if lines else np.zeros(valid.shape, bool))
    return seam_steps(x, valid, built, T, lines, tile) + dark_patches(x, valid, built, T, emask, tile)


def stretches(cands):
    """Contiguous stretches of stepped seam, one row each: length, mean step, and whether it is long
    enough (STRETCH_M) to be a cloud shadow cut by the seam."""
    steps = cands[cands["kind"] == "seam_step"]
    rows = []
    for tile, d in steps.groupby("tile"):
        # overlapping substrings of one seam do not node exactly, so contiguity is taken from 1 m wide
        # buffers: each connected piece is a stretch, its length ~ area / 1 m
        pieces = shapely.union_all(list(d.geometry.buffer(0.5)))
        for part in getattr(pieces, "geoms", [pieces]):
            near = d[d.geometry.intersects(part)]
            length = part.area / 1.0
            rows.append({"tile": tile, "length_m": float(length), "score": float(near["score"].mean()),
                         "shadow": bool(length >= STRETCH_M),
                         "geometry": shapely.MultiLineString(list(near.geometry))})
    cols = ["tile", "length_m", "score", "shadow", "geometry"]
    return gpd.GeoDataFrame(rows, columns=cols, geometry="geometry", crs=CRS)


def find_shadows(tiles, edges, use_bdtopo=True):
    rows = []
    for t in tiles:
        rows += tile_shadows(t, list(edges), use_bdtopo)
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=CRS) if rows else \
        gpd.GeoDataFrame(columns=["tile", "kind", "score", "geometry"], geometry="geometry", crs=CRS)


def run(name, use_bdtopo):
    out = OUT / name
    cfg = load_campaigns()[name]
    edges = gpd.read_file(out / "edges.gpkg").geometry
    c = find_shadows(cfg["tiles"], edges, use_bdtopo)
    pts = gpd.read_file(out / "artefacts.gpkg")
    sh = pts[(pts["art_arttype_id"] == OMBRE) & (pts["in_tiles"] == 1)]
    pos_tiles = {str(int(a)) for a in sh["art_id"]}
    hit = c.groupby("tile")["score"].max()
    lines = [f"# Cloud shadows (experimental): {name}\n\n{len(c)} candidates "
             f"({(c['kind'] == 'seam_step').sum()} seam steps, {(c['kind'] == 'dark_patch').sum()} dark patches) "
             f"in {c['tile'].nunique()} of {len(cfg['tiles'])} tiles.\n",
             f"\nTiles with a reported shadow: {sorted(pos_tiles)}; with a candidate: "
             f"{sorted(pos_tiles & set(hit.index))}.\n",
             f"\nOther tiles with a candidate: {len(set(hit.index) - pos_tiles)} "
             f"({sorted(set(hit.index) - pos_tiles)}).\n"]
    st = stretches(c)
    flagged = set(st.loc[st["shadow"], "tile"])
    lines.append(f"\nSeam stretches ≥ {STRETCH_M:.0f} m (the decision): tiles {sorted(flagged)}; "
                 f"reported-shadow tiles found {len(flagged & pos_tiles)}/{len(pos_tiles)}, other tiles flagged "
                 f"{len(flagged - pos_tiles)}. Calibrated on these same examples (in-sample).\n")
    c.to_file(out / "cloud_candidates.gpkg")
    st.to_file(out / "cloud_candidates.gpkg", layer="stretches")
    (out / "cloud_report.md").write_text("".join(lines))
    print("".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--campaign")
    ap.add_argument("--no-bdtopo", action="store_true")
    args = ap.parse_args()
    for name in load_campaigns():
        if args.campaign in (None, name):
            run(name, not args.no_bdtopo)


if __name__ == "__main__":
    main()
