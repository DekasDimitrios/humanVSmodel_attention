import numpy as np
import pytest

from hvma.heatmaps import (argmax_point, center_prior, inpaint_tokens, normalize, outlier_tokens,
                           points_to_heatmap, resize, texture_energy, threshold_topk,
                           to_distribution, to_grid)
from PIL import Image


def test_normalize_range_and_constant():
    hm = np.array([[1.0, 3.0], [2.0, 5.0]], dtype=np.float32)
    out = normalize(hm)
    assert out.min() == 0.0 and out.max() == 1.0
    assert np.all(normalize(np.ones((4, 4))) == 0)


def test_to_distribution_sums_to_one_and_handles_zero():
    assert np.isclose(to_distribution(np.random.rand(5, 5)).sum(), 1.0)
    flat = to_distribution(np.zeros((3, 3)))
    assert np.allclose(flat, 1 / 9)


def test_points_to_heatmap_peak_at_point():
    hm = points_to_heatmap([(0.25, 0.75)], size=(100, 100), sigma_frac=0.03)
    assert hm.shape == (100, 100) and hm.dtype == np.float32
    x, y = argmax_point(hm)
    assert abs(x - 0.25) < 0.03 and abs(y - 0.75) < 0.03


def test_points_to_heatmap_ignores_out_of_bounds_and_weights():
    empty = points_to_heatmap([(1.5, 0.5)], size=(20, 20))
    assert empty.sum() == 0
    hm = points_to_heatmap([(0.2, 0.2), (0.8, 0.8)], size=(50, 50), weights=[0.1, 1.0])
    x, y = argmax_point(hm)
    assert x > 0.5 and y > 0.5  # heavier point wins
    with pytest.raises(ValueError):
        points_to_heatmap([(0.5, 0.5)], size=(10, 10), weights=[1.0, 1.0])


def test_threshold_topk_fraction():
    hm = np.random.rand(40, 40)
    mask = threshold_topk(hm, 0.1)
    assert abs(mask.mean() - 0.1) < 0.01
    assert threshold_topk(hm, 1.0).all()
    with pytest.raises(ValueError):
        threshold_topk(hm, 0.0)


def test_resize_shape_and_values():
    hm = np.zeros((7, 7), dtype=np.float32)
    hm[3, 3] = 1.0
    out = resize(hm, (70, 70))
    assert out.shape == (70, 70)
    x, y = argmax_point(out)
    assert abs(x - 0.5) < 0.1 and abs(y - 0.5) < 0.1


def test_center_prior_peaks_in_the_middle():
    hm = center_prior((41, 61), sigma_frac=0.2)
    assert hm.shape == (41, 61) and hm.max() == 1.0
    x, y = argmax_point(hm)
    assert abs(x - 0.5) < 0.02 and abs(y - 0.5) < 0.03


def test_outlier_tokens_and_inpainting():
    norms = np.ones((5, 5))
    norms[1, 3] = 20.0  # one high-norm token
    mask = outlier_tokens(norms, ratio=2.0)
    assert mask.sum() == 1 and mask[1, 3]
    grid = np.full((5, 5), 0.2, dtype=np.float32)
    grid[0, 0] = 1.0  # keeps the range non-degenerate after normalising
    grid[1, 3] = 50.0  # the outlier's hot spot dominates the raw map
    out = inpaint_tokens(grid, mask)
    assert out[1, 3] == out[2, 2]  # in-painted from its (uniform) neighbourhood
    assert out[0, 0] == 1.0  # untouched cells keep their relative values
    assert np.array_equal(inpaint_tokens(grid, np.zeros_like(mask)), normalize(grid))


def test_to_grid_and_texture_energy():
    hm = np.zeros((70, 70), dtype=np.float32)
    hm[:35, :35] = 1.0
    g = to_grid(hm, 2)
    assert g.shape == (2, 2) and g[0, 0] > 0.9 and g[1, 1] < 0.1
    img = np.zeros((224, 224), dtype=np.uint8)
    img[:, 112:] = (np.arange(112) // 2 % 2) * 255  # 2-px stripes (1-px ones have no Sobel gradient)
    tex = texture_energy(Image.fromarray(img).convert("RGB"), g=2)
    assert tex[:, 1].min() > 10 * tex[:, 0].max()  # detail on the right, flat on the left
