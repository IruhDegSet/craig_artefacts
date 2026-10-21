"""Hand-made cross-seam features for every station (the interpretable baseline's input).

    python features.py [--campaign NAME] [--workers 8]

Writes out/<campaign>/features.parquet (one row per station, same order as stations.parquet).

Each station is read as a 256 px seam-aligned crop at 5 cm (12.8 m along the seam). For every crop
row the seam column is located, and the pixels just left/right of it are compared:
  - profiles a few pixels off each side of the seam should match if the mosaic is seamless;
    a cut or offset object breaks that match, and a shifted copy (best lag) may restore it
  - a radiometric seam shows as a colour jump between the two bands
  - a visible cut shows as a strong gradient on the seam compared with just beside it
"""

import argparse
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd

from config import OUT, load_campaigns
from crops import CropReader

SIZE, CROP_GSD = 256, 0.05
NEAR = (3, 9)      # px off the seam for the profiles
BAND = 40          # px (2 m) each side for colour statistics
MAX_LAG = 40       # px (2 m) searched along the seam
SEARCH = 48        # px around the crop centre where the seam is looked for in each row


def seam_columns(side):
    """Column of the seam in each row (left->right sign change near the centre), -1 if none."""
    c = side.shape[1] // 2
    win = side[:, c - SEARCH:c + SEARCH]
    hit = (win[:, :-1] == 1) & (win[:, 1:] <= 0)
    col = np.where(hit.any(1), hit.argmax(1) + c - SEARCH, -1)
    return col


def profiles(gray, side, valid, cols):
    """Mean of NEAR px left and right of the seam in each row, and the rows where both are usable."""
    h, w = gray.shape
    rows = np.flatnonzero(cols >= 0)
    ks = np.arange(*NEAR)
    lc = np.clip(cols[rows, None] - ks[None] - 1, 0, w - 1)
    rc = np.clip(cols[rows, None] + ks[None] + 1, 0, w - 1)
    ok = ((side[rows[:, None], lc] == 1).all(1) & (side[rows[:, None], rc] == -1).all(1)
          & valid[rows[:, None], lc].all(1) & valid[rows[:, None], rc].all(1))
    rows, lc, rc = rows[ok], lc[ok], rc[ok]
    return rows, gray[rows[:, None], lc].mean(1), gray[rows[:, None], rc].mean(1)


def ncc(a, b):
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 1e-6 else 0.0


def lag_search(gl, gr):
    """Best normalised correlation of gr shifted along the seam, and its lag in px."""
    best, lag = -1.0, 0
    n = len(gl)
    for d in range(-MAX_LAG, MAX_LAG + 1, 2):
        a, b = (gl[d:], gr[:n - d]) if d >= 0 else (gl[:n + d], gr[-d:])
        if len(a) < 32:
            continue
        v = ncc(a, b)
        if v > best:
            best, lag = v, d
    return best, lag


def crop_features(c):
    rgb, side, valid, dist = c["rgb"], c["side"], c["valid"], c["dist"]
    f = {"valid_frac": float(valid.mean())}
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gmag = np.hypot(gx, gy)

    bl = (side == 1) & valid & (dist <= BAND)
    br = (side == -1) & valid & (dist <= BAND)
    if bl.sum() > 200 and br.sum() > 200:
        f["dE_lab"] = float(np.linalg.norm(lab[bl].mean(0) - lab[br].mean(0)))
        f["dL"] = float(lab[bl, 0].mean() - lab[br, 0].mean())
        f["std_ratio"] = float(np.log((gray[bl].std() + 1) / (gray[br].std() + 1)))
        f["texture"] = float(gmag[bl | br].mean())
        sl, sr = hsv[bl, 1], hsv[br, 1]
        f["sat_p98_max"] = float(max(np.percentile(sl, 98), np.percentile(sr, 98)))
        f["sat_p98_diff"] = float(abs(np.percentile(sl, 98) - np.percentile(sr, 98)))
        f["bright_p99_diff"] = float(abs(np.percentile(gray[bl], 99) - np.percentile(gray[br], 99)))

    cols = seam_columns(side)
    f["seam_rows"] = float((cols >= 0).mean())
    for tag, sl in (("", slice(None)), ("_c", slice(SIZE // 4, 3 * SIZE // 4))):
        rows, pl, pr = profiles(gray[sl], side[sl], valid[sl], cols[sl])
        if len(rows) < 40:
            continue
        diff = pl - pr
        f["prof_corr" + tag] = ncc(pl, pr)
        f["prof_mad" + tag] = float(np.abs(diff - np.median(diff)).mean() / (pl.std() + pr.std() + 1))
        f["prof_absdiff_p90" + tag] = float(np.percentile(np.abs(diff - np.median(diff)), 90))
        gl, gr = np.diff(pl), np.diff(pr)
        c0 = ncc(gl, gr)
        best, lag = lag_search(gl, gr)
        f["grad_corr0" + tag] = c0
        f["grad_best" + tag] = best
        f["grad_gain" + tag] = best - c0
        f["grad_lag" + tag] = float(abs(lag))

    # a cut is a strong edge exactly on the seam compared with just beside it
    rows = np.flatnonzero(cols >= 0)
    if len(rows) > 40:
        cc = cols[rows]
        at = np.abs(gray[rows, np.clip(cc + 2, 0, SIZE - 1)] - gray[rows, np.clip(cc - 2, 0, SIZE - 1)])
        bes = (np.abs(gray[rows, np.clip(cc + 12, 0, SIZE - 1)] - gray[rows, np.clip(cc + 8, 0, SIZE - 1)])
               + np.abs(gray[rows, np.clip(cc - 8, 0, SIZE - 1)] - gray[rows, np.clip(cc - 12, 0, SIZE - 1)])) / 2
        f["seam_jump"] = float(at.mean() / (bes.mean() + 1))
        f["seam_jump_p90"] = float(np.percentile(at - bes, 90))
    return f


_reader = None


def _init(out_dir):
    global _reader
    _reader = CropReader(out_dir)


def _one(row):
    x, y, angle = row
    return crop_features(_reader.crop(x, y, angle, SIZE, CROP_GSD))


def build(name, workers):
    out = OUT / name
    st = pd.read_parquet(out / "stations.parquet")
    rows = list(st[["x", "y", "angle"]].itertuples(index=False, name=None))
    with Pool(workers, initializer=_init, initargs=(out,)) as pool:
        feats = pool.map(_one, rows, chunksize=64)
    df = pd.DataFrame(feats, index=st.index)
    df.to_parquet(out / "features.parquet")
    print(f"{name}: {df.shape[1]} features for {len(df)} stations, "
          f"missing: {df.isna().mean().round(2)[df.isna().mean() > 0].to_dict()}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--campaign")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    for name in load_campaigns():
        if args.campaign in (None, name):
            build(name, args.workers)


if __name__ == "__main__":
    main()
