"""Shared constants and campaign loading."""

import glob
import os
from pathlib import Path

import yaml

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


def load_campaigns(path=None):
    """{name: cfg} with tile paths expanded and other paths made absolute.

    The file is campaigns.yaml unless the SEAM_QA_CAMPAIGNS environment variable names another one."""
    path = Path(path or os.environ.get("SEAM_QA_CAMPAIGNS", ROOT / "campaigns.yaml"))
    cfgs = yaml.safe_load(path.read_text())
    for name, c in cfgs.items():
        tiles = []
        for pattern in c["tiles"]:
            tiles += glob.glob(str(path.parent / pattern), recursive=True)
        c["tiles"] = sorted(set(tiles))
        c["seamlines"] = str((path.parent / c["seamlines"]).resolve())
        c["artefacts"] = str((path.parent / c["artefacts"]).resolve())
        c.setdefault("campaign", name)
        c.setdefault("image_field", "Image")
    return cfgs
