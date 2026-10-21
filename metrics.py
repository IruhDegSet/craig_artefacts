"""Point-level evaluation of station scores, as an operator would use them: a ranked review list.

Scores are turned into detections by non-maximum suppression (NMS_RADIUS m). An artefact instance
is found when a detection lies within MATCH_RADIUS m of one of its points. Budgets are expressed
in detections per km of seam, so they transfer between campaigns.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from config import MATCH_RADIUS, SEAM_TYPES, STRIDE

NMS_RADIUS = 8.0
BUDGETS = (1, 2, 5, 10, 20)  # detections per km of seam


def detections(xy, score):
    """Indices of the stations kept by NMS, best first."""
    order = np.argsort(-score, kind="stable")
    tree = cKDTree(xy)
    taken = np.zeros(len(xy), bool)
    keep = []
    for i in order:
        if taken[i]:
            continue
        keep.append(i)
        taken[tree.query_ball_point(xy[i], NMS_RADIUS)] = True
    return np.array(keep, int)


def targets(artefacts_path):
    """Seam-type artefact instances inside the tiles: instance id, type, and the xy of their points."""
    pts = gpd.read_file(artefacts_path)
    pts = pts[(pts["in_tiles"] == 1) & (pts["is_seam_type"] == 1)]
    return pts


def found_ranks(st, score, pts):
    """For each artefact instance, the rank of the first detection that finds it (inf if never).

    Also returns the instance ids and, per ranked detection, the set of instances it hits."""
    xy = st[["x", "y"]].to_numpy()
    det = detections(xy, np.asarray(score, float))
    inst = pts["instance"].to_numpy()
    ids = np.unique(inst)
    pxy = np.c_[pts.geometry.x, pts.geometry.y]
    found_at = {i: np.inf for i in ids}
    d, j = cKDTree(pxy).query(xy[det], k=min(8, len(pxy)), distance_upper_bound=MATCH_RADIUS)
    d, j = np.atleast_2d(d), np.atleast_2d(j)
    hit_inst = []
    for rank in range(len(det)):
        hits = {inst[jj] for dd, jj in zip(d[rank], j[rank]) if np.isfinite(dd)}
        hit_inst.append(hits)
        for h in hits:
            found_at[h] = min(found_at[h], rank)
    return ids, np.array([found_at[i] for i in ids]), hit_inst


def operating_points(st, score, pts, budgets=(0.25, 0.5, 1, 2, 5, 10)):
    """What fully automatic flagging would give if it kept the top `budget` detections per km: the score
    threshold that implies, precision (share of detections within MATCH_RADIUS of a seam-type point)
    and instance recall. Scores are not calibrated across models, hence budgets rather than fixed
    thresholds. Precision is a lower bound when only sent-back artefacts are labelled."""
    xy = st[["x", "y"]].to_numpy()
    score = np.asarray(score, float)
    det = detections(xy, score)
    km = len(st) * STRIDE / 1000
    pxy = np.c_[pts.geometry.x, pts.geometry.y]
    near = cKDTree(pxy).query(xy[det], distance_upper_bound=MATCH_RADIUS)[0] <= MATCH_RADIUS
    _, ranks, _ = found_ranks(st, score, pts)
    rows = []
    for b in budgets:
        n = min(len(det), max(1, round(b * km)))
        rows.append({"per_km": b, "detections": n, "threshold": round(float(score[det[n - 1]]), 3),
                     "precision": round(float(near[:n].mean()), 2), "recall": round(float((ranks < n).mean()), 2)})
    return pd.DataFrame(rows)


def threshold_table(st, score, pts, thresholds=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95)):
    """Automatic flagging at absolute score thresholds: detections (after NMS) per km, precision (share
    within MATCH_RADIUS of a seam-type point, a lower bound) and instance recall."""
    xy = st[["x", "y"]].to_numpy()
    score = np.asarray(score, float)
    det = detections(xy, score)
    km = len(st) * STRIDE / 1000
    pxy = np.c_[pts.geometry.x, pts.geometry.y]
    near = cKDTree(pxy).query(xy[det], distance_upper_bound=MATCH_RADIUS)[0] <= MATCH_RADIUS
    _, ranks, _ = found_ranks(st, score, pts)
    rows = []
    for t in thresholds:
        n = int((score[det] >= t).sum())  # det is sorted by score, so these are the first n
        rows.append({"threshold": t, "per_km": round(n / km, 2),
                     "precision": round(float(near[:n].mean()), 2) if n else None,
                     "recall": round(float((ranks < n).mean()), 2)})
    return rows


def evaluate(st, score, pts, on_seam_only=False):
    """dict of recall per budget, per-type recall at 5/km, and instance-level average precision."""
    if on_seam_only:
        pts = pts[pts["d_seam_line"] <= 5.0]
    km = len(st) * STRIDE / 1000
    ids, ranks, hit_inst = found_ranks(st, score, pts)
    res = {"km": km, "n_instances": len(ids)}
    for b in BUDGETS:
        res[f"R@{b}/km"] = float((ranks < round(b * km)).mean())
    types = pts.groupby("instance")["art_arttype_id"].first()
    for t, name in SEAM_TYPES.items():
        sel = np.array([types[i] == t for i in ids])
        if sel.any():
            res[f"R@5/km {name}"] = f"{int((ranks[sel] < round(5 * km)).sum())}/{int(sel.sum())}"
    # AP: walk the ranked detections, a detection is a TP if it finds a not-yet-found instance
    seen, tp = set(), []
    for hits in hit_inst:
        new = hits - seen
        tp.append(1 if new else 0)
        seen |= hits
    tp = np.array(tp)
    prec = np.cumsum(tp) / np.arange(1, len(tp) + 1)
    res["AP"] = float((prec * tp).sum() / len(ids))
    return res
