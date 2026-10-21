"""Seam-aligned crops: the seam runs vertically through the centre, walking up the crop follows the
seam tangent, the photo on the seam's left (img_L) is on the crop's left.

    reader = CropReader(out_dir)                      # out/<campaign>/ from seam_graph.py
    c = reader.crop(x, y, angle, size=256, gsd=0.05)  # dict of arrays, see crop()
"""

import cv2
import geopandas as gpd
import numpy as np
import rasterio
import shapely
from rasterio.enums import Resampling
from rasterio.windows import from_bounds


class CropReader:
    def __init__(self, out_dir):
        self.ds = rasterio.open(out_dir / "mosaic.vrt")
        self.native = abs(self.ds.transform.a)
        edges = gpd.read_file(out_dir / "edges.gpkg")
        self.lines = edges.geometry.values
        self.tree = shapely.STRtree(self.lines)

    def crop(self, x, y, angle, size=256, gsd=0.05):
        """
        rgb    (size, size, 3) uint8
        valid  (size, size) bool, inside the mosaic (alpha > 0)
        side   (size, size) int8, +1 left photo, -1 right photo, 0 another photo or the seam itself
        dist   (size, size) float32, distance to the nearest seam in pixels
        """
        c = (size - 1) / 2
        r = size * gsd / 2 * 1.5  # covers the rotated square
        n = int(np.ceil(2 * r / gsd))
        x0, y1 = x - r, y + r
        win = from_bounds(x0, y1 - n * gsd, x0 + n * gsd, y1, self.ds.transform)
        resampling = Resampling.average if gsd > 1.5 * self.native else Resampling.bilinear
        src = self.ds.read(window=win, out_shape=(self.ds.count, n, n), boundless=True, fill_value=0,
                           resampling=resampling)
        src = np.ascontiguousarray(src.transpose(1, 2, 0))

        t = np.array([np.cos(angle), np.sin(angle)])
        eu = np.array([np.sin(angle), -np.cos(angle)])  # -normal: crop right = right of the seam
        ev = -t                                         # crop down = backwards along the seam
        M = np.array([
            # the source is read at the output resolution, so crop pixel -> source pixel is a rotation
            [eu[0], ev[0], (x - x0) / gsd - c * (eu[0] + ev[0])],
            [-eu[1], -ev[1], (y1 - y) / gsd + c * (eu[1] + ev[1])],
        ], np.float32)
        out = cv2.warpAffine(src, M, (size, size), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        rgb = out[..., :3]
        valid = out[..., 3] > 0 if out.shape[2] > 3 else np.ones((size, size), bool)

        seam = np.zeros((size, size), np.uint8)
        half = size * gsd * 0.75
        for i in self.tree.query(shapely.box(x - half, y - half, x + half, y + half)):
            xy = shapely.get_coordinates(self.lines[i]) - (x, y)
            uv = np.c_[c + xy @ eu / gsd, c + xy @ ev / gsd]
            cv2.polylines(seam, [np.round(uv * 4).astype(np.int32)], False, 255, 1, cv2.LINE_8, shift=2)
        dist = cv2.distanceTransform(255 - seam, cv2.DIST_L2, 3)
        _, comp = cv2.connectedComponents((seam == 0).astype(np.uint8), connectivity=4)
        side = np.zeros((size, size), np.int8)
        ci, off = int(round(c)), max(3, size // 32)
        left, right = comp[ci, ci - off], comp[ci, ci + off]
        if left:
            side[comp == left] = 1
        if right and right != left:
            side[comp == right] = -1
        return {"rgb": rgb, "valid": valid, "side": side, "dist": dist}

    def close(self):
        self.ds.close()
