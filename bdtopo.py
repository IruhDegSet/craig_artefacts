"""BD TOPO buildings and roads around a tile, from the IGN Geoplateforme WFS.

Same paging/retry/cache scheme as REVET-OBSTACLE/data_prep/footprints.py:fetch_block, but any
layer, queried in EPSG:2154 and cached per tile under cache/bdtopo/<layer>/<tile>.geojson.
"""

import json
import time

import geopandas as gpd
import requests
from pyproj import Transformer

from config import CACHE, CRS

WFS_URL = "https://data.geopf.fr/wfs/ows"
PAGE = 5000
BUILDINGS = "BDTOPO_V3:batiment"
ROADS = "BDTOPO_V3:troncon_de_route"
# objects that are not on the orthophoto
EXCLUDED_STATES = {"En projet", "Non réalisé", "Détruit"}
DEFAULT_HEIGHT = 6.0  # m, when hauteur is missing
DEFAULT_ROAD_WIDTH = 5.0  # m, when largeur_de_chaussee is missing

_to_lonlat = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True).transform


def fetch(layer, bounds, key):
    """Features of `layer` intersecting bounds (EPSG:2154), paginated and cached under `key`."""
    # the position makes the key unique: tile names repeat across campaigns
    cache_file = CACHE / "bdtopo" / layer.split(":")[1] / f"{key}_{bounds[0]:.0f}_{bounds[1]:.0f}.geojson"
    if cache_file.exists():
        return json.loads(cache_file.read_text())["features"]
    lon0, lat0 = _to_lonlat(bounds[0], bounds[1])
    lon1, lat1 = _to_lonlat(bounds[2], bounds[3])
    features, start = [], 0
    while True:
        params = {
            "SERVICE": "WFS", "VERSION": "2.0.0", "REQUEST": "GetFeature",
            "TYPENAMES": layer, "OUTPUTFORMAT": "application/json",
            "SRSNAME": CRS, "COUNT": PAGE, "STARTINDEX": start,
            # WFS 2.0 with an EPSG URN takes lat/lon axis order
            "BBOX": f"{lat0},{lon0},{lat1},{lon1},urn:ogc:def:crs:EPSG::4326",
        }
        for attempt in range(5):
            try:
                r = requests.get(WFS_URL, params=params, timeout=120)
                r.raise_for_status()
                page = r.json()
                break
            except (requests.RequestException, ValueError):
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
        features.extend(page["features"])
        if len(page["features"]) < PAGE:
            break
        start += PAGE
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    return features


def _frame(features, columns):
    feats = [f for f in features
             if f.get("geometry") and f["properties"].get("etat_de_l_objet") not in EXCLUDED_STATES]
    if not feats:
        return gpd.GeoDataFrame(columns=columns + ["geometry"], geometry="geometry", crs=CRS)
    gdf = gpd.GeoDataFrame.from_features(feats, crs=CRS)
    gdf["geometry"] = gdf.geometry.force_2d()
    for c in columns:
        if c not in gdf:
            gdf[c] = None
    return gdf[columns + ["geometry"]]


def buildings(bounds, key):
    gdf = _frame(fetch(BUILDINGS, bounds, key), ["hauteur", "nature"])
    gdf["hauteur"] = gdf["hauteur"].astype(float).fillna(DEFAULT_HEIGHT)
    gdf["geometry"] = gdf.geometry.make_valid()
    return gdf


def roads(bounds, key):
    gdf = _frame(fetch(ROADS, bounds, key), ["largeur_de_chaussee", "nature", "importance"])
    gdf["largeur_de_chaussee"] = gdf["largeur_de_chaussee"].astype(float).fillna(DEFAULT_ROAD_WIDTH)
    return gdf
