import numpy as np
import pytest

from hvma.heatmaps import points_to_heatmap
from hvma.metrics import (cc, compute_all, disagreement_score, iou_topk, js_divergence, nss,
                          pointing_game, spearman)


@pytest.fixture
def maps():
    same = points_to_heatmap([(0.3, 0.3)], size=(64, 64), sigma_frac=0.05)
    other = points_to_heatmap([(0.8, 0.8)], size=(64, 64), sigma_frac=0.05)
    return same, other


def test_identical_maps_score_perfectly(maps):
    same, _ = maps
    assert cc(same, same) == pytest.approx(1.0)
    assert spearman(same, same) == pytest.approx(1.0)
    assert iou_topk(same, same, 0.2) == pytest.approx(1.0)
    assert pointing_game(same, same) == 1.0
    assert js_divergence(same, same) == pytest.approx(0.0, abs=1e-9)


def test_disjoint_maps_score_badly(maps):
    same, other = maps
    assert cc(same, other) < 0.0
    assert iou_topk(same, other, 0.1) == 0.0
    assert pointing_game(same, other) == 0.0
    assert js_divergence(same, other) > 0.5


def test_nss_high_at_fixations_low_elsewhere(maps):
    same, _ = maps
    assert nss(same, [(0.3, 0.3)]) > 2.0
    assert nss(same, [(0.9, 0.9)]) < 0.0
    assert np.isnan(nss(same, []))


def test_shape_mismatch_raises(maps):
    same, _ = maps
    with pytest.raises(ValueError):
        cc(same, same[:32, :32])


def test_constant_map_gives_nan_not_crash():
    flat = np.zeros((16, 16), dtype=np.float32)
    bump = points_to_heatmap([(0.5, 0.5)], size=(16, 16))
    assert np.isnan(cc(flat, bump))
    assert np.isnan(pointing_game(flat, bump))


def test_compute_all_keys_and_disagreement_order(maps):
    same, other = maps
    good = compute_all(same, same, points=[(0.3, 0.3)])
    bad = compute_all(same, other, points=[(0.3, 0.3)])
    for k in ("cc", "spearman", "js_divergence", "pointing_game", "iou@10", "iou@20", "iou@30", "nss"):
        assert k in good
    assert disagreement_score(good) < disagreement_score(bad)
    assert 0.0 <= disagreement_score(good) <= 1.0
