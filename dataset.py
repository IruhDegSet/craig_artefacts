"""Two-scale seam-aligned crops for the deep model, cached once per campaign.

    python dataset.py [--campaign NAME] [--workers 8]

Writes out/<campaign>/crops_<SIZE>.npy (uint8, N x 2 scales x SIZE x SIZE x 5 channels: RGB, side, dist)
so training does not re-read the mosaic. Scale 0 is 5 cm (9.6 m of seam), scale 1 is 10 cm (19.2 m);
both are centred on the station and share its orientation. For campaigns too big to cache, use
CropReader directly in the Dataset (same arrays, read on the fly).
"""

import argparse
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch

from config import OUT, load_campaigns
from crops import CropReader

SIZE = 192
SCALES = (0.05, 0.10)
DIST_CLIP = 32  # px
MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255
STD = np.array([0.229, 0.224, 0.225], np.float32) * 255


def read_station(reader, x, y, angle):
    """(scales, SIZE, SIZE, 5) uint8 crop of one station: RGB, side code, distance."""
    out = np.zeros((len(SCALES), SIZE, SIZE, 5), np.uint8)
    for k, gsd in enumerate(SCALES):
        c = reader.crop(x, y, angle, SIZE, gsd)
        rgb = c["rgb"] * c["valid"][..., None]
        out[k, ..., :3] = rgb
        out[k, ..., 3] = (c["side"] + 1).astype(np.uint8)  # 0 right, 1 other/seam, 2 left
        out[k, ..., 4] = np.minimum(c["dist"], DIST_CLIP).astype(np.uint8)
    return out


_reader = None


def _init(out_dir):
    global _reader
    _reader = CropReader(out_dir)


def _one(row):
    return read_station(_reader, *row)


class Source:
    """The crops of one campaign, by station index: from the cache when it exists, else read from
    the mosaic. File handles open lazily, so a Source can be sent to DataLoader workers."""

    def __init__(self, name):
        self.name = name
        self.out = OUT / name
        self.cache = self.out / f"crops_{SIZE}.npy"
        self._arr = self._reader = self._rows = None

    def __getstate__(self):
        return {"name": self.name, "out": self.out, "cache": self.cache,
                "_arr": None, "_reader": None, "_rows": None}

    def __getitem__(self, i):
        if self.cache.exists():
            if self._arr is None:
                self._arr = np.load(self.cache, mmap_mode="r")
            return np.array(self._arr[i])
        if self._reader is None:
            self._reader = CropReader(self.out)
            self._rows = pd.read_parquet(self.out / "stations.parquet")[["x", "y", "angle"]].to_numpy()
        return read_station(self._reader, *self._rows[i])


def iter_crops(name, idx, workers=8, bs=64):
    """Crops of stations `idx` of a campaign, in order, as arrays of up to `bs`."""
    out = OUT / name
    cache = out / f"crops_{SIZE}.npy"
    if cache.exists():
        arr = np.load(cache, mmap_mode="r")
        for s in range(0, len(idx), bs):
            yield np.array(arr[np.sort(idx[s:s + bs])])[np.argsort(np.argsort(idx[s:s + bs]))]
        return
    rows = pd.read_parquet(out / "stations.parquet")[["x", "y", "angle"]].to_numpy()[idx]
    with Pool(workers, initializer=_init, initargs=(out,)) as pool:
        it = pool.imap(_one, map(tuple, rows), chunksize=16)
        for s in range(0, len(idx), bs):
            yield np.stack([next(it) for _ in range(min(bs, len(idx) - s))])


def cache(name, workers):
    out = OUT / name
    st = pd.read_parquet(out / "stations.parquet")
    arr = np.lib.format.open_memmap(out / f"crops_{SIZE}.npy", mode="w+", dtype=np.uint8,
                                    shape=(len(st), len(SCALES), SIZE, SIZE, 5))
    rows = list(st[["x", "y", "angle"]].itertuples(index=False, name=None))
    with Pool(workers, initializer=_init, initargs=(out,)) as pool:
        for i, a in enumerate(pool.imap(_one, rows, chunksize=32)):
            arr[i] = a
    arr.flush()
    print(f"{name}: cached {arr.shape} -> {out / f'crops_{SIZE}.npy'}")


def to_tensor(crop):
    """(scales, SIZE, SIZE, 5) uint8 -> (scales, 4, SIZE, SIZE) float: normalised RGB + signed distance."""
    rgb = (crop[..., :3].astype(np.float32) - MEAN) / STD
    side = crop[..., 3].astype(np.float32) - 1
    sdist = side * crop[..., 4].astype(np.float32) / DIST_CLIP
    x = np.concatenate([rgb, sdist[..., None]], -1).transpose(0, 3, 1, 2)
    return torch.from_numpy(np.ascontiguousarray(x))


def flip(crop, rng):
    """Random flips. Along the seam (rows) is free; across it swaps the sides, so the side code flips."""
    if rng.random() < 0.5:
        crop = crop[:, ::-1]
    if rng.random() < 0.5:
        crop = crop[:, :, ::-1].copy()
        crop[..., 3] = 2 - crop[..., 3]
    return np.ascontiguousarray(crop)


def jitter(crop, rng):
    """Brightness/contrast jitter on RGB, same for both scales."""
    a, b = rng.uniform(0.85, 1.15), rng.uniform(-15, 15)
    crop = crop.copy()
    crop[..., :3] = np.clip(crop[..., :3].astype(np.float32) * a + b, 0, 255).astype(np.uint8)
    return crop


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--campaign")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    for name in load_campaigns():
        if args.campaign in (None, name):
            cache(name, args.workers)


if __name__ == "__main__":
    main()
