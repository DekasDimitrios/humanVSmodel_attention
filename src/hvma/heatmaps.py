"""Heatmap utilities. Pure numpy/scipy; everything here is unit-tested.

Conventions (see CLAUDE.md): heatmaps are float32 (H, W) in [0, 1]; points are normalised
(x, y) in [0, 1] with origin top-left.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

Point = tuple[float, float]


def normalize(hm: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Max-normalise to [0, 1]. A constant map becomes all zeros rather than NaN."""
    hm = np.asarray(hm, dtype=np.float32)
    hm = hm - hm.min()
    mx = hm.max()
    return hm / mx if mx > eps else np.zeros_like(hm)


def to_distribution(hm: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Sum-normalise to a probability map (used by divergence metrics)."""
    hm = np.clip(np.asarray(hm, dtype=np.float64), 0, None)
    s = hm.sum()
    if s <= eps:
        return np.full(hm.shape, 1.0 / hm.size)
    return hm / s


def resize(hm: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Bilinear resize to `size=(H, W)`. Uses PIL so we don't pull in OpenCV just for this."""
    h, w = size
    img = Image.fromarray(np.asarray(hm, dtype=np.float32), mode="F")
    out = np.asarray(img.resize((w, h), resample=Image.BILINEAR), dtype=np.float32)
    return out


def points_to_heatmap(
    points: Iterable[Point],
    size: tuple[int, int],
    sigma_frac: float = 0.04,
    weights: Sequence[float] | None = None,
) -> np.ndarray:
    """Splat normalised points onto an (H, W) grid and blur with a gaussian.

    `sigma_frac` is relative to image width so the blur is resolution-independent.
    `weights` lets you weight each click by the labeller's reliability (Rapidata `userScore`).
    """
    h, w = size
    canvas = np.zeros((h, w), dtype=np.float32)
    pts = list(points)
    if weights is None:
        weights = [1.0] * len(pts)
    if len(weights) != len(pts):
        raise ValueError("weights must match points")
    for (x, y), wt in zip(pts, weights):
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            continue  # a tap just outside the image; ignore rather than crash
        col = min(int(round(x * (w - 1))), w - 1)
        row = min(int(round(y * (h - 1))), h - 1)
        canvas[row, col] += float(wt)
    if canvas.sum() == 0:
        return canvas
    sigma = max(sigma_frac * w, 0.5)
    return normalize(gaussian_filter(canvas, sigma=sigma))


def threshold_topk(hm: np.ndarray, frac: float) -> np.ndarray:
    """Boolean mask of the top `frac` fraction of pixels by value."""
    if not 0 < frac <= 1:
        raise ValueError("frac must be in (0, 1]")
    flat = np.asarray(hm, dtype=np.float32).ravel()
    k = max(1, int(round(frac * flat.size)))
    if k >= flat.size:
        return np.ones(hm.shape, dtype=bool)
    thresh = np.partition(flat, flat.size - k)[flat.size - k]
    arr = np.asarray(hm)
    if thresh <= 0:
        # Sparse map (e.g. a gaussian whose tails were truncated to exact 0): the "top k%" is
        # just its support. Without this, the tie at 0 would select the entire image.
        return arr > 0
    return arr >= thresh


def argmax_point(hm: np.ndarray) -> Point:
    """Location of the maximum as a normalised (x, y) point."""
    h, w = hm.shape
    r, c = np.unravel_index(int(np.argmax(hm)), hm.shape)
    return (c / max(w - 1, 1), r / max(h - 1, 1))


def to_uint8(hm: np.ndarray) -> np.ndarray:
    return (np.clip(normalize(hm), 0, 1) * 255).astype(np.uint8)
