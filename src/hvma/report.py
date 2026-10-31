"""Evaluation + reporting: metrics CSV, overlay PNGs, summary figures, agreement.json for the viewer."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .config import Config
from .data import Sample, load_human_heatmap, load_image, load_manifest, load_points
from .heatmaps import normalize, resize, to_uint8
from .metrics import compute_all, disagreement_score

METHODS = ("gradcam", "rollout")


# --------------------------------------------------------------------------- saliency I/O


def saliency_path(cfg: Config, sample: Sample, method: str) -> Path:
    return cfg.paths.saliency_dir / f"{sample.id}.{method}.npy"


def load_saliency(cfg: Config, sample: Sample, method: str) -> np.ndarray | None:
    p = saliency_path(cfg, sample, method)
    return np.load(p) if p.exists() else None


# --------------------------------------------------------------------------- evaluate


def evaluate(cfg: Config, human_source: str) -> list[dict]:
    """One row per (sample, method) with all metrics. Writes results/metrics_<source>.csv."""
    samples = load_manifest(cfg)
    rows: list[dict] = []
    for s in samples:
        human = load_human_heatmap(cfg, s, human_source)
        if human is None:
            continue
        points = load_points(cfg, s, human_source)
        for method in METHODS:
            sal = load_saliency(cfg, s, method)
            if sal is None:
                continue
            model = normalize(resize(sal, human.shape))
            m = compute_all(human, model, points=points, topk_fracs=cfg.heatmap.topk_fracs)
            row = {"id": s.id, "method": method, "human_source": human_source,
                   "prompt": s.prompt, "coherence_score": s.coherence_score, **m}
            row["disagreement"] = disagreement_score(row)
            rows.append(row)
    if not rows:
        raise RuntimeError(
            f"nothing to evaluate for human_source={human_source!r}: "
            "need both human heatmaps and saliency maps (run `hvma fetch`/`parse-results` and `hvma saliency`)"
        )
    out = cfg.paths.results_dir / f"metrics_{human_source}.csv"
    _write_csv(out, rows)
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def summarize(rows: list[dict]) -> list[dict]:
    """Mean of each numeric metric per method. Written to results/summary.csv."""
    out: list[dict] = []
    for method in METHODS:
        sub = [r for r in rows if r["method"] == method]
        if not sub:
            continue
        agg: dict = {"method": method, "n": len(sub)}
        for k in sub[0]:
            if k in ("id", "method", "human_source", "prompt"):
                continue
            vals = np.array([r[k] for r in sub if r.get(k) is not None], dtype=float)
            vals = vals[~np.isnan(vals)]
            agg[k] = float(vals.mean()) if vals.size else float("nan")
        out.append(agg)
    return out


# --------------------------------------------------------------------------- figures


def _colorize(hm: np.ndarray) -> Image.Image:
    """Grey heatmap -> RGBA magma-like overlay without matplotlib (keeps this fast)."""
    v = to_uint8(hm).astype(np.float32) / 255.0
    r = np.clip(1.5 * v, 0, 1)
    g = np.clip(1.5 * v - 0.5, 0, 1) ** 1.5
    b = np.clip(0.3 + 0.7 * (1 - v) * v * 4, 0, 1) * (v > 0.05)
    a = np.clip(v * 1.2, 0, 0.85)
    rgba = np.stack([r, g, b, a], axis=-1)
    return Image.fromarray((rgba * 255).astype(np.uint8), mode="RGBA")


def overlay(img: Image.Image, hm: np.ndarray) -> Image.Image:
    base = img.convert("RGBA")
    layer = _colorize(resize(hm, (img.height, img.width)))
    return Image.alpha_composite(base, layer).convert("RGB")


def write_overlays(cfg: Config, human_source: str) -> None:
    samples = load_manifest(cfg)
    out_dir = cfg.paths.overlays_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for s in samples:
        img = load_image(s)
        human = load_human_heatmap(cfg, s, human_source)
        if human is not None:
            overlay(img, human).save(out_dir / f"{s.id}.human_{human_source}.jpg", quality=85)
        for method in METHODS:
            sal = load_saliency(cfg, s, method)
            if sal is not None:
                overlay(img, sal).save(out_dir / f"{s.id}.{method}.jpg", quality=85)


def disagreement_gallery(cfg: Config, rows: list[dict], human_source: str,
                         method: str = "gradcam", k: int = 6) -> Path:
    """Top-k most disagreeing images: image | human | model, one row each."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    samples = {s.id: s for s in load_manifest(cfg)}
    sub = sorted((r for r in rows if r["method"] == method),
                 key=lambda r: -r["disagreement"])[:k]
    fig, axes = plt.subplots(len(sub), 3, figsize=(9, 3 * len(sub)))
    axes = np.atleast_2d(axes)
    for i, r in enumerate(sub):
        s = samples[r["id"]]
        img = load_image(s)
        human = load_human_heatmap(cfg, s, human_source)
        sal = load_saliency(cfg, s, method)
        axes[i, 0].imshow(img); axes[i, 0].set_title(f"{s.id}  cc={r['cc']:.2f}  iou20={r['iou@20']:.2f}", fontsize=8)
        axes[i, 1].imshow(overlay(img, human)); axes[i, 1].set_title("humans", fontsize=8)
        axes[i, 2].imshow(overlay(img, sal)); axes[i, 2].set_title(f"model ({method})", fontsize=8)
        for ax in axes[i]:
            ax.axis("off")
    fig.tight_layout()
    cfg.paths.figures_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.paths.figures_dir / "disagreement_gallery.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def metric_histograms(cfg: Config, rows: list[dict]) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = ["cc", "iou@20", "pointing_game", "js_divergence"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 3.2))
    for ax, m in zip(axes, metrics):
        for method in METHODS:
            vals = np.array([r[m] for r in rows if r["method"] == method], dtype=float)
            vals = vals[~np.isnan(vals)]
            if vals.size:
                ax.hist(vals, bins=12, alpha=0.55, label=f"{method} (μ={vals.mean():.2f})")
        ax.set_title(m, fontsize=10)
        ax.legend(fontsize=7)
    fig.tight_layout()
    cfg.paths.figures_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.paths.figures_dir / "metric_histograms.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- viewer payload


def write_agreement_json(cfg: Config, rows: list[dict], human_source: str) -> Path:
    """results/agreement.json: what the viewer reads. Paths are relative to results/."""
    samples = {s.id: s for s in load_manifest(cfg)}
    by_id: dict[str, dict] = {}
    for r in rows:
        s = samples[r["id"]]
        entry = by_id.setdefault(r["id"], {
            "id": s.id,
            "prompt": s.prompt,
            "coherence_score": s.coherence_score,
            "image": str(Path(s.image_path).as_posix()),
            "human_overlay": f"overlays/{s.id}.human_{human_source}.jpg",
            "methods": {},
        })
        entry["methods"][r["method"]] = {
            "overlay": f"overlays/{s.id}.{r['method']}.jpg",
            **{k: (None if isinstance(v, float) and np.isnan(v) else v)
               for k, v in r.items() if k not in ("id", "method", "human_source", "prompt", "coherence_score")},
        }
    payload = {"human_source": human_source, "samples": sorted(
        by_id.values(),
        key=lambda e: -max(m["disagreement"] for m in e["methods"].values()))}
    out = cfg.paths.results_dir / "agreement.json"
    with out.open("w") as fh:
        json.dump(payload, fh, indent=1)
    return out
