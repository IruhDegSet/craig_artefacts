"""Shared constants and campaign loading."""

import glob
import os
import subprocess
from pathlib import Path

import rasterio
import yaml
from rasterio.enums import ColorInterp

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
CACHE = ROOT / "cache"
CRS = "EPSG:2154"

# art_arttype_id of the artefacts caused by the seamline (measured: all within ~2 m of a seam).
# Hotspot (10), Ombre de nuage (14) and types 3/5/15/21 sit 20-80 m away and are not seam errors.
SEAM_TYPES = {2: "batiment_coupe", 8: "vehicule_coupe", 13: "decalage", 16: "erreur_mosaiquage"}
TYPE_NAMES = {**SEAM_TYPES, 10: "hotspot", 14: "ombre_de_nuage"}

STRIDE = 4.0          # m between stations along a seam
TANGENT_HALF = 3.0    # m on each side of a station used to estimate the seam direction
SIDE_OFFSET = 1.5     # m off the seam to decide which photo is on each side
DEDUPE = 2.0          # m: a station closer than this to one of another tile is the same station

POS_RADIUS = 3.0      # m from a seam-class point: positive station
IGNORE_RADIUS = 30.0  # m from a seam-class point: not used as a negative
OTHER_IGNORE = 10.0   # m from any other artefact point: not used as a negative
MATCH_RADIUS = 10.0   # m: a detection this close to an artefact finds it (evaluation and MIL bags)
MERGE_RADIUS = 20.0   # m: same-type points closer than this are one artefact (339/340, 3788/3789, ...)

# Relief displacement mismatch across a seam is height * baseline / H, H = flying height.
# 5 cm GSD with a large-format camera (f ~100 mm, 4 um pixels) gives H ~ 1250 m.
FLYING_HEIGHT = 1250.0
GSD = 0.05


def to_l93(tiles, vrt_dir):
    """The tiles with every one not in EPSG:2154 replaced by a warped VRT in Lambert-93.

    Everything downstream (mosaic.vrt, BD TOPO, artefact points) is in EPSG:2154. A VRT is a few KB:
    the pixels are resampled when read, at the tile's own resolution, so no reprojected copy is
    written. The VRT keeps the tile's file name (tile ids are file stems) under a folder named after
    the tile's own folder (the same id can sit in two class folders). An alpha band is added when
    the tile has none, so the warped borders are transparent rather than black."""
    out = []
    for t in tiles:
        with rasterio.open(t) as ds:
            if ds.crs is not None and ds.crs.to_epsg() == 2154:
                out.append(t)
                continue
            if ds.crs is None:
                raise SystemExit(f"{t}: no CRS, cannot reproject it to EPSG:2154")
            res, has_alpha = ds.res[0], ColorInterp.alpha in ds.colorinterp
        vrt = Path(vrt_dir) / Path(t).parent.name / (Path(t).stem + ".vrt")
        if not vrt.exists():
            vrt.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["gdalwarp", "-q", "-of", "VRT", "-t_srs", CRS, "-tr", str(res), str(res), "-tap",
                            "-r", "bilinear", *([] if has_alpha else ["-dstalpha"]), str(t), str(vrt)], check=True)
        out.append(str(vrt))
    return out


def load_campaigns(path=None):
    """{name: cfg} with tile paths expanded and other paths made absolute.

    `seamlines` is one file or a list of files (e.g. a block delivered in several parts). Tiles that
    are not in Lambert-93 (lot 3 is in CC47) are read through warped VRTs under cache/l93/<campaign>/.
    The file is campaigns.yaml unless the SEAM_QA_CAMPAIGNS environment variable names another one."""
    path = Path(path or os.environ.get("SEAM_QA_CAMPAIGNS", ROOT / "campaigns.yaml"))
    cfgs = yaml.safe_load(path.read_text())
    for name, c in cfgs.items():
        tiles = []
        for pattern in c["tiles"]:
            tiles += glob.glob(str(path.parent / pattern), recursive=True)
        c["tiles"] = to_l93(sorted(set(tiles)), CACHE / "l93" / name)
        seamlines = c["seamlines"] if isinstance(c["seamlines"], list) else [c["seamlines"]]
        c["seamlines"] = [str((path.parent / s).resolve()) for s in seamlines]
        c["artefacts"] = str((path.parent / c["artefacts"]).resolve())
        c.setdefault("campaign", name)
        c.setdefault("image_field", "Image")
    return cfgs
