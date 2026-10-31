import numpy as np

from hvma.data import dataset_heatmap_to_array


def test_dataset_heatmap_real_encoding():
    """The HF dataset stores `coherence_heatmap` as a 2-D list of float16 at image resolution
    (1024x1024 in practice), with values well below 1 and absent for ~30% of rows."""
    raw = np.zeros((8, 8), dtype=np.float16)
    raw[2, 5] = 0.16
    out = dataset_heatmap_to_array(raw.tolist(), (8, 8))
    assert out.shape == (8, 8) and out.dtype == np.float32
    assert out.max() == 1.0 and np.unravel_index(out.argmax(), out.shape) == (2, 5)
    assert dataset_heatmap_to_array(raw.tolist(), (16, 16)).shape == (16, 16)
    assert dataset_heatmap_to_array(None, (8, 8)) is None
