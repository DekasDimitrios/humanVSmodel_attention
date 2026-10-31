"""Agreement metrics between a human attention map and a model saliency map.

All functions take (human, model) float32 maps of the *same* shape and return a float.
Resize the model map to the human map's shape first (see `heatmaps.resize`).

Metric cheat-sheet (standard saliency-evaluation metrics, see Bylinskii et al. 2018):
- pointing_game: 1 if the model's peak falls inside the human top-k region, else 0.
- iou_topk:      overlap of the two top-k% regions. 0..1.
- cc:            Pearson correlation of the two maps. -1..1.
- spearman:      rank correlation; robust to monotone rescaling. -1..1.
- nss:           model map z-scored, averaged at human fixation points. Higher is better;
                 ~0 means chance. Needs the raw human points.
- js_divergence: symmetric divergence between the two maps as distributions. 0 = identical.
- border_mass:   not an agreement metric — the share of the *model* map's mass in the outer
                 band of the image (see `border_mass`). Quantifies the edge bias.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr, spearmanr

from .heatmaps import Point, argmax_point, threshold_topk, to_distribution


def _check(a: np.ndarray, b: np.ndarray) -> None:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}; resize first")
    if a.ndim != 2:
        raise ValueError("heatmaps must be 2-D")


def pointing_game(human: np.ndarray, model: np.ndarray, frac: float = 0.2) -> float:
    _check(human, model)
    if human.max() <= 0:
        return float("nan")
    mask = threshold_topk(human, frac)
    x, y = argmax_point(model)
    h, w = human.shape
    r = min(int(round(y * (h - 1))), h - 1)
    c = min(int(round(x * (w - 1))), w - 1)
    return float(mask[r, c])


def iou_topk(human: np.ndarray, model: np.ndarray, frac: float = 0.2) -> float:
    _check(human, model)
    a = threshold_topk(human, frac)
    b = threshold_topk(model, frac)
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else float("nan")


def cc(human: np.ndarray, model: np.ndarray) -> float:
    _check(human, model)
    h, m = human.ravel(), model.ravel()
    if h.std() == 0 or m.std() == 0:
        return float("nan")
    return float(pearsonr(h, m)[0])


def spearman(human: np.ndarray, model: np.ndarray) -> float:
    _check(human, model)
    h, m = human.ravel(), model.ravel()
    if h.std() == 0 or m.std() == 0:
        return float("nan")
    return float(spearmanr(h, m)[0])


def nss(model: np.ndarray, points: Iterable[Point]) -> float:
    """Normalised Scanpath Saliency: mean z-scored model value at human click locations."""
    pts = [p for p in points if 0 <= p[0] <= 1 and 0 <= p[1] <= 1]
    if not pts or model.std() == 0:
        return float("nan")
    z = (model - model.mean()) / model.std()
    h, w = model.shape
    vals = []
    for x, y in pts:
        r = min(int(round(y * (h - 1))), h - 1)
        c = min(int(round(x * (w - 1))), w - 1)
        vals.append(z[r, c])
    return float(np.mean(vals))


def js_divergence(human: np.ndarray, model: np.ndarray) -> float:
    _check(human, model)
    p = to_distribution(human).ravel()
    q = to_distribution(model).ravel()
    # scipy returns the JS *distance* (sqrt of divergence); square it to get the divergence in bits.
    return float(jensenshannon(p, q, base=2) ** 2)


BORDER_FRAC = 1 / 7  # two patches of a 14x14 ViT grid


def border_mass(hm: np.ndarray, frac: float = BORDER_FRAC) -> float:
    """Share of the map's mass within `frac` of any image edge.

    A uniform map scores the band's area, 1 - (1 - 2*frac)^2 (0.49 for frac=1/7).
    """
    p = to_distribution(hm)
    h, w = p.shape
    bh, bw = max(1, round(frac * h)), max(1, round(frac * w))
    return float(1.0 - p[bh:h - bh, bw:w - bw].sum())


def compute_all(
    human: np.ndarray,
    model: np.ndarray,
    points: Iterable[Point] | None = None,
    topk_fracs: Iterable[float] = (0.1, 0.2, 0.3),
) -> dict[str, float]:
    """Every metric in one dict; keys are stable so they become CSV columns."""
    out: dict[str, float] = {
        "cc": cc(human, model),
        "spearman": spearman(human, model),
        "js_divergence": js_divergence(human, model),
        "pointing_game": pointing_game(human, model, frac=0.2),
        "border_mass": border_mass(model),
    }
    for f in topk_fracs:
        out[f"iou@{int(round(f * 100))}"] = iou_topk(human, model, frac=f)
    if points is not None:
        out["nss"] = nss(model, points)
    return out


def disagreement_score(row: dict[str, float]) -> float:
    """Single number for sorting the viewer: low correlation + low overlap = high disagreement.

    Uses cc and iou@20 because both are bounded and interpretable; NaNs count as maximal
    disagreement so broken cases float to the top where a human will look at them.
    """
    c = row.get("cc", float("nan"))
    i = row.get("iou@20", float("nan"))
    c = -1.0 if np.isnan(c) else c
    i = 0.0 if np.isnan(i) else i
    return float((1 - c) / 2 + (1 - i)) / 2
