"""Synthetic seam artefacts on clean seam-aligned crops (both scales at once, geometry in metres).

Every artefact changes ONE side of the seam only, which is what a mosaicking error is: the two
photos disagree. Types (index = type head class):
  0 object_cut   a blob straddling the seam is displaced on one side   (bâtiment coupé)
  1 vehicle_cut  a vehicle straddling the seam exists on one side only (véhicule coupé)
  2 shift        the whole side is offset along/across the seam        (décalage, erreur de mosaïquage)
  3 radiometric  gain/offset jump between the two sides                (raccord radiométrique)

Generalises augment_cut_shift / misalign_image_by_cut from ~/Downloads/CRAIG (random cut lines
in a patch) to the real seam of each crop.
"""

import cv2
import numpy as np

from dataset import SCALES, SIZE

TYPES = ["batiment_coupe", "vehicule_coupe", "decalage", "radiometrie"]
REAL_TYPE = {2: 0, 8: 1, 13: 2, 16: 2}  # art_arttype_id -> type head class
CAR_COLOURS = np.array([[235, 235, 235], [25, 25, 30], [130, 135, 140], [170, 30, 30], [30, 60, 140],
                        [200, 200, 205], [90, 90, 95]], np.float32)


def _px(m, gsd):
    return m / gsd


def _side_mask(crop_k, side):
    return crop_k[..., 3] == side


def _shift(img, du, dv):
    M = np.float32([[1, 0, du], [0, 1, dv]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)


def object_cut(crop, rng):
    side = rng.choice([0, 2])
    a, b = rng.uniform(2, 8, 2)                 # blob half-axes, m
    cu = rng.uniform(-a / 2, a / 2)             # across the seam, so it straddles it
    cv = rng.uniform(-1.5, 1.5)                 # along the seam
    ang = rng.uniform(0, 180)
    mag, th = rng.uniform(0.3, 1.5), rng.uniform(0, 2 * np.pi)
    for k, g in enumerate(SCALES):
        c = (SIZE - 1) / 2
        blob = np.zeros((SIZE, SIZE), np.uint8)
        cv2.ellipse(blob, (int(c + _px(cu, g)), int(c + _px(cv, g))),
                    (max(1, int(_px(a, g))), max(1, int(_px(b, g)))), ang, 0, 360, 1, -1)
        m = (blob > 0) & _side_mask(crop[k], side)
        moved = _shift(crop[k, ..., :3], _px(mag * np.cos(th), g), _px(mag * np.sin(th), g))
        crop[k, ..., :3][m] = moved[m]
    return crop


def vehicle_cut(crop, rng):
    side = rng.choice([0, 2])
    w, l = rng.uniform(1.6, 2.0), rng.uniform(3.8, 5.0)
    ang = rng.uniform(0, 180)
    cu = rng.uniform(-l / 3, l / 3)
    cv = rng.uniform(-1.5, 1.5)
    colour = CAR_COLOURS[rng.integers(len(CAR_COLOURS))] + rng.normal(0, 10, 3)
    shadow_dir = rng.uniform(0, 2 * np.pi)
    for k, g in enumerate(SCALES):
        c = (SIZE - 1) / 2
        centre = (c + _px(cu, g), c + _px(cv, g))
        img = crop[k, ..., :3].astype(np.float32)
        layer = img.copy()
        # shadow, then body, then darker windscreen and rear window bands
        sh = ((centre[0] + _px(0.5 * np.cos(shadow_dir), g), centre[1] + _px(0.5 * np.sin(shadow_dir), g)),
              (_px(0.9 * w, g), _px(0.9 * l, g)), ang)
        shadow = np.zeros(img.shape[:2], np.float32)
        cv2.fillPoly(shadow, [cv2.boxPoints(sh).astype(np.int32)], 1.0)
        shadow = cv2.GaussianBlur(shadow, (0, 0), max(0.5, 0.15 / g))[..., None]
        layer = img * (1 - 0.45 * shadow)
        body = (centre, (_px(w, g), _px(l, g)), ang)
        cv2.fillPoly(layer, [cv2.boxPoints(body).astype(np.int32)], colour.tolist())
        d = np.array([-np.sin(np.radians(ang)), np.cos(np.radians(ang))])  # long axis in pixels
        for off, frac in ((0.22, 0.18), (-0.30, 0.12)):
            p = (centre[0] + d[0] * _px(off * l, g), centre[1] + d[1] * _px(off * l, g))
            band = (p, (_px(0.85 * w, g), _px(frac * l, g)), ang)
            cv2.fillPoly(layer, [cv2.boxPoints(band).astype(np.int32)], (colour * 0.25).tolist())
        layer = cv2.GaussianBlur(layer, (0, 0), max(0.4, 0.03 / g))
        m = _side_mask(crop[k], side)
        crop[k, ..., :3][m] = np.clip(layer[m], 0, 255).astype(np.uint8)
    return crop


def shift(crop, rng):
    side = rng.choice([0, 2])
    dv = rng.choice([-1, 1]) * rng.uniform(0.15, 2.0)
    du = rng.uniform(-0.3, 0.3)
    for k, g in enumerate(SCALES):
        m = _side_mask(crop[k], side)
        moved = _shift(crop[k, ..., :3], _px(du, g), _px(dv, g))
        crop[k, ..., :3][m] = moved[m]
    return crop


def radiometric(crop, rng):
    side = rng.choice([0, 2])
    # real raccords are subtle: a few % per channel, at least one channel off by ~6-10 %
    gain = rng.uniform(0.95, 1.05, 3)
    gain[rng.integers(3)] = rng.choice([-1, 1]) * rng.uniform(0.06, 0.10) + 1
    offset = rng.uniform(-6, 6)
    for k in range(len(SCALES)):
        m = _side_mask(crop[k], side)
        crop[k, ..., :3][m] = np.clip(crop[k, ..., :3][m] * gain + offset, 0, 255).astype(np.uint8)
    return crop


MAKERS = [object_cut, vehicle_cut, shift, radiometric]
WEIGHTS = np.array([0.3, 0.3, 0.25, 0.15])


def make(crop, rng):
    """A synthetic positive from a clean crop: (crop, type index)."""
    t = rng.choice(len(MAKERS), p=WEIGHTS)
    return MAKERS[t](crop.copy(), rng), t
