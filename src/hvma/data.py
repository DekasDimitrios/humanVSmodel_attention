"""Stream samples from Rapidata's Hugging Face dataset and keep a local manifest.

The dataset (`Rapidata/text-2-image-Rich-Human-Feedback`) is ~17 GB, so we never download it;
we stream, shuffle with a buffer, and take the first N. Each kept sample is written to
`data/images/<id>.png` and described in `data/manifest.json`.

The `coherence_heatmap` column is the human "tap on artefacts" signal. Its exact encoding was
not verified when this file was written — `dataset_heatmap_to_array` is deliberately tolerant
(flat list, nested list, numpy array, any resolution) and `hvma inspect-dataset` prints the raw
form so the function can be tightened once seen.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from PIL import Image

from .config import Config
from .heatmaps import normalize, resize


@dataclass
class Sample:
    id: str
    image_path: str
    prompt: str
    width: int
    height: int
    coherence_score: float | None = None
    alignment_score: float | None = None
    style_score: float | None = None
    has_dataset_heatmap: bool = False
    # Filled in when a live Locate job runs; lets us join results back to the sample.
    live_datapoint_ref: str | None = None


def stream_rows(cfg: Config, shuffle: bool = True) -> Iterator[dict[str, Any]]:
    """Yield rows from the HF dataset in streaming mode (lazy imports keep CLI startup fast).

    Each `coherence_heatmap` is a 1024x1024 float16 grid. `IterableDataset.shuffle()` (at any
    buffer size) loses the Arrow->numpy fast path and peaks at ~10 GB RAM, so instead we shuffle
    the *order of the parquet shards* with the seed and stream them unshuffled in numpy format
    (~3 GB peak). `fetch` adds randomness within a shard by subsampling with `keep_prob`.
    """
    from datasets import load_dataset
    from huggingface_hub import HfApi

    d = cfg.dataset
    shards = sorted(
        f for f in HfApi().list_repo_files(d.name, repo_type="dataset")
        if f.startswith(f"data/{d.split}-") and f.endswith(".parquet")
    )
    if not shards:
        raise FileNotFoundError(f"no data/{d.split}-*.parquet shards in {d.name}")
    if shuffle:
        shards = [shards[i] for i in np.random.default_rng(d.seed).permutation(len(shards))]
    ds = load_dataset(d.name, data_files={d.split: shards}, split=d.split, streaming=True)
    yield from ds.with_format("numpy")


def _score(row: dict[str, Any], key: str) -> float | None:
    v = row.get(key)
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def dataset_heatmap_to_array(raw: Any, size: tuple[int, int]) -> np.ndarray | None:
    """Best-effort conversion of the dataset's heatmap field to a (H, W) float32 map in [0, 1].

    Handles: None/empty -> None; 2-D list/array -> resized; flat list of n*n values -> reshaped
    to square then resized; flat list of H*W matching `size` -> reshaped directly.
    """
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=np.float32)
    if arr.size == 0:
        return None
    h, w = size
    if arr.ndim == 1:
        if arr.size == h * w:
            arr = arr.reshape(h, w)
        else:
            side = int(round(np.sqrt(arr.size)))
            if side * side != arr.size:
                raise ValueError(
                    f"Flat heatmap of length {arr.size} is neither H*W={h*w} nor square; "
                    "inspect the dataset and fix dataset_heatmap_to_array()."
                )
            arr = arr.reshape(side, side)
    elif arr.ndim == 3 and arr.shape[-1] in (1, 3):
        arr = arr.mean(axis=-1)
    elif arr.ndim != 2:
        raise ValueError(f"Unexpected heatmap ndim={arr.ndim}, shape={arr.shape}")
    if arr.shape != (h, w):
        arr = resize(arr, (h, w))
    return normalize(arr)


def fetch(cfg: Config, n: int | None = None) -> list[Sample]:
    """Stream `n` samples, save images + dataset heatmaps, write the manifest."""
    n = n or cfg.dataset.n_samples
    dcfg = cfg.dataset
    paths = cfg.paths
    paths.images_dir.mkdir(parents=True, exist_ok=True)
    human_dir = paths.human_dir / "dataset"
    human_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(dcfg.seed)

    samples: list[Sample] = []
    for row in stream_rows(cfg):
        if len(samples) >= n:
            break
        if rng.random() > dcfg.keep_prob:
            continue
        coh = _score(row, "coherence_score_norm")
        if (dcfg.max_coherence_score is not None and coh is not None
                and coh > dcfg.max_coherence_score):
            continue
        raw_hm = row.get(dcfg.heatmap_column)
        if dcfg.require_heatmap and (raw_hm is None or np.asarray(raw_hm).size == 0):
            continue
        img = row["image"]
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)  # numpy format decodes images to HxWxC uint8
        elif not isinstance(img, Image.Image):
            img = Image.open(img)  # datasets may hand back a file-like object
        img = img.convert("RGB")
        sid = f"s{len(samples):04d}"
        img_path = paths.images_dir / f"{sid}.png"
        img.save(img_path)

        hm = dataset_heatmap_to_array(raw_hm, (img.height, img.width))
        if hm is not None:
            np.save(human_dir / f"{sid}.npy", hm)

        samples.append(
            Sample(
                id=sid,
                image_path=str(img_path),
                prompt=str(row.get("prompt", "")),
                width=img.width,
                height=img.height,
                coherence_score=coh,
                alignment_score=_score(row, "alignment_score_norm"),
                style_score=_score(row, "style_score_norm"),
                has_dataset_heatmap=hm is not None,
            )
        )
    save_manifest(cfg, samples)
    return samples


def inspect_first_row(cfg: Config) -> dict[str, Any]:
    """Return type/shape info for every column of the first streamed row (no images saved)."""
    row = next(iter(stream_rows(cfg, shuffle=False)))
    info: dict[str, Any] = {}
    for k, v in row.items():
        if isinstance(v, Image.Image):
            info[k] = f"PIL.Image {v.size} {v.mode}"
        elif isinstance(v, (list, tuple, np.ndarray)):
            arr = np.asarray(v)
            info[k] = f"{type(v).__name__} shape={arr.shape} dtype={arr.dtype} " \
                      f"min={arr.min() if arr.size else None} max={arr.max() if arr.size else None}"
        else:
            s = str(v)
            info[k] = f"{type(v).__name__}: {s[:120]}{'…' if len(s) > 120 else ''}"
    return info


def save_manifest(cfg: Config, samples: list[Sample]) -> None:
    cfg.paths.manifest.parent.mkdir(parents=True, exist_ok=True)
    with cfg.paths.manifest.open("w") as fh:
        json.dump([asdict(s) for s in samples], fh, indent=2)


def load_manifest(cfg: Config) -> list[Sample]:
    if not cfg.paths.manifest.exists():
        raise FileNotFoundError(f"{cfg.paths.manifest} not found — run `hvma fetch` first")
    with cfg.paths.manifest.open() as fh:
        return [Sample(**d) for d in json.load(fh)]


def load_image(sample: Sample) -> Image.Image:
    return Image.open(sample.image_path).convert("RGB")


def load_human_heatmap(cfg: Config, sample: Sample, source: str) -> np.ndarray | None:
    """`source` is 'dataset' or 'live'."""
    p = cfg.paths.human_dir / source / f"{sample.id}.npy"
    return np.load(p) if p.exists() else None


def load_points(cfg: Config, sample: Sample, source: str = "live") -> list[tuple[float, float]] | None:
    p = cfg.paths.human_dir / source / f"{sample.id}.points.json"
    if not p.exists():
        return None
    with p.open() as fh:
        return [tuple(pt) for pt in json.load(fh)]  # type: ignore[misc]
