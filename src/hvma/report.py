"""Evaluation + reporting: metrics CSV, overlay PNGs, summary figures, agreement.json for the viewer.

Methods come in two groups: the *main* methods (`model.methods`, incl. the `center` baseline)
go to results/summary.csv; the edge-bias *ablation* (`ablation.methods()`) goes to
results/ablation.csv and its own figures, with the main maps repeated there for reference.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .config import Config
from .data import Sample, load_human_heatmap, load_image, load_manifest, load_points
from .heatmaps import center_prior, normalize, resize, to_uint8
from .metrics import BORDER_FRAC, border_mass, compute_all, disagreement_score

BASELINES = ("center",)  # model-free maps, computed on the fly rather than loaded from disk

# Reference palette (dataviz skill, categorical slots 1-3) and chart ink.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"


def main_methods(cfg: Config) -> list[str]:
    return list(cfg.model.methods)


def ablation_methods(cfg: Config) -> list[str]:
    return cfg.ablation.methods()


def all_methods(cfg: Config) -> list[str]:
    return main_methods(cfg) + [m for m in ablation_methods(cfg) if m not in cfg.model.methods]


def group_of(cfg: Config, method: str) -> str:
    return "main" if method in cfg.model.methods else "ablation"


# --------------------------------------------------------------------------- saliency I/O


def saliency_path(cfg: Config, sample: Sample, method: str) -> Path:
    return cfg.paths.saliency_dir / f"{sample.id}.{method}.npy"


def load_saliency(cfg: Config, sample: Sample, method: str) -> np.ndarray | None:
    p = saliency_path(cfg, sample, method)
    return np.load(p) if p.exists() else None


def model_map(cfg: Config, sample: Sample, method: str, shape: tuple[int, int]) -> np.ndarray | None:
    """The `method` map for `sample` at `shape`, in [0, 1]; None if it was never computed."""
    if method == "center":
        return center_prior(shape, cfg.model.center_sigma_frac)
    sal = load_saliency(cfg, sample, method)
    return None if sal is None else normalize(resize(sal, shape))


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
        for method in all_methods(cfg):
            model = model_map(cfg, s, method, human.shape)
            if model is None:
                continue
            m = compute_all(human, model, points=points, topk_fracs=cfg.heatmap.topk_fracs)
            row = {"id": s.id, "method": method, "group": group_of(cfg, method),
                   "human_source": human_source, "prompt": s.prompt,
                   "coherence_score": s.coherence_score, **m}
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


def summarize(rows: list[dict], methods: list[str]) -> list[dict]:
    """Mean of each numeric metric per method, in the order of `methods`."""
    out: list[dict] = []
    for method in methods:
        sub = [r for r in rows if r["method"] == method]
        if not sub:
            continue
        agg: dict = {"method": method, "group": sub[0]["group"], "n": len(sub)}
        for k in sub[0]:
            if k in ("id", "method", "group", "human_source", "prompt"):
                continue
            vals = np.array([r[k] for r in sub if r.get(k) is not None], dtype=float)
            vals = vals[~np.isnan(vals)]
            agg[k] = float(vals.mean()) if vals.size else float("nan")
        out.append(agg)
    return out


def human_border_mass(cfg: Config, human_source: str) -> float:
    """Mean border mass of the human maps: where the edge-bias numbers should land."""
    vals = [border_mass(h) for s in load_manifest(cfg)
            if (h := load_human_heatmap(cfg, s, human_source)) is not None]
    return float(np.mean(vals)) if vals else float("nan")


def ablation_summary(cfg: Config, rows: list[dict], human_source: str) -> list[dict]:
    """Ablation table: a human reference row, the main maps, then every ablation variant."""
    reference = [m for m in ("center", "gradcam", "rollout") if m in all_methods(cfg)]
    methods = reference + [m for m in ablation_methods(cfg) if m not in reference]
    human = {"method": "human (reference)", "group": "reference",
             "border_mass": human_border_mass(cfg, human_source)}
    return [human] + summarize(rows, methods)


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
        for method in all_methods(cfg):
            hm = model_map(cfg, s, method, (img.height, img.width))
            if hm is not None:
                overlay(img, hm).save(out_dir / f"{s.id}.{method}.jpg", quality=85)


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _top_disagreeing(rows: list[dict], method: str, k: int) -> list[dict]:
    return sorted((r for r in rows if r["method"] == method), key=lambda r: -r["disagreement"])[:k]


def disagreement_gallery(cfg: Config, rows: list[dict], human_source: str,
                         method: str = "gradcam", k: int = 6) -> Path:
    """Top-k most disagreeing images: image | human | model, one row each."""
    plt = _plt()
    samples = {s.id: s for s in load_manifest(cfg)}
    sub = _top_disagreeing(rows, method, k)
    fig, axes = plt.subplots(len(sub), 3, figsize=(9, 3 * len(sub)))
    axes = np.atleast_2d(axes)
    for i, r in enumerate(sub):
        s = samples[r["id"]]
        img = load_image(s)
        human = load_human_heatmap(cfg, s, human_source)
        hm = model_map(cfg, s, method, (img.height, img.width))
        axes[i, 0].imshow(img); axes[i, 0].set_title(f"{s.id}  cc={r['cc']:.2f}  iou20={r['iou@20']:.2f}", fontsize=8)
        axes[i, 1].imshow(overlay(img, human)); axes[i, 1].set_title("humans", fontsize=8)
        axes[i, 2].imshow(overlay(img, hm)); axes[i, 2].set_title(f"model ({method})", fontsize=8)
        for ax in axes[i]:
            ax.axis("off")
    fig.tight_layout()
    cfg.paths.figures_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.paths.figures_dir / "disagreement_gallery.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def metric_histograms(cfg: Config, rows: list[dict]) -> Path:
    """Distribution of each metric for the main methods (ablations: see ablation_chart)."""
    plt = _plt()
    metrics = ["cc", "iou@20", "pointing_game", "js_divergence"]
    colors = dict(zip(main_methods(cfg), (BLUE, ORANGE, AQUA)))
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 3.2))
    for ax, m in zip(axes, metrics):
        for method in main_methods(cfg):
            vals = np.array([r[m] for r in rows if r["method"] == method], dtype=float)
            vals = vals[~np.isnan(vals)]
            if vals.size:
                ax.hist(vals, bins=12, alpha=0.55, color=colors.get(method),
                        label=f"{method} (μ={vals.mean():.2f})")
        ax.set_title(m, fontsize=10)
        ax.legend(fontsize=7)
    fig.tight_layout()
    cfg.paths.figures_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.paths.figures_dir / "metric_histograms.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def ablation_chart(cfg: Config, summary: list[dict]) -> Path:
    """Two panels sharing the method axis: border mass (the edge bias) and cc with humans.

    Separate panels rather than two y-scales; reference values are drawn as labelled lines.
    """
    plt = _plt()
    from matplotlib.patches import Patch

    human = next(r for r in summary if r["group"] == "reference")
    rows = [r for r in summary if r["group"] != "reference"][::-1]  # top-to-bottom = table order
    names = [r["method"] for r in rows]
    colors = [BLUE if r["group"] == "main" else ORANGE for r in rows]
    y = np.arange(len(rows))
    uniform = 1 - (1 - 2 * BORDER_FRAC) ** 2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 0.42 * len(rows) + 1.6), sharey=True)
    for ax in (ax1, ax2):
        ax.set_facecolor("#fcfcfb")
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#c3c2b7")
        ax.tick_params(colors=INK_2, labelsize=9, length=0)

    ax1.barh(y, [r["border_mass"] for r in rows], color=colors, height=0.62)
    for x, label, style in ((uniform, "uniform map", "--"),
                            (human["border_mass"], "humans", ":")):
        ax1.axvline(x, color=MUTED, linestyle=style, linewidth=1.2)
        ax1.text(x, len(rows) - 0.35, f" {label} {x:.2f}", color=INK_2, fontsize=8, va="bottom")
    ax1.set_xlim(0, 1)
    ax1.set_title("Share of saliency in the outer border (lower = less edge bias)",
                  fontsize=10, color=INK, loc="left")

    ax2.barh(y, [r["cc"] for r in rows], color=colors, height=0.62)
    ax2.axvline(0, color="#c3c2b7", linewidth=1)
    ax2.set_title("Correlation with human taps (cc)", fontsize=10, color=INK, loc="left")

    ax1.set_yticks(y, names)
    ax1.set_ylim(-0.6, len(rows) + 0.4)
    fig.legend(handles=[Patch(color=BLUE, label="main method"),
                        Patch(color=ORANGE, label="ablation variant")],
               loc="lower center", ncol=2, frameon=False, fontsize=9, labelcolor=INK_2)
    fig.patch.set_facecolor("#fcfcfb")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    cfg.paths.figures_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.paths.figures_dir / "ablation_edge_bias.png"
    fig.savefig(out, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def ablation_gallery(cfg: Config, rows: list[dict], human_source: str, k: int = 4) -> Path:
    """The k images where gradcam disagrees most, shown under every ablation variant."""
    plt = _plt()
    samples = {s.id: s for s in load_manifest(cfg)}
    methods: list[str] = ["gradcam"]
    for m in ablation_methods(cfg):
        methods += ["rollout", m] if m == "rollout_masked" else [m]  # pair each with its source
    methods = [m for m in dict.fromkeys(methods + ["rollout", "center"]) if m in all_methods(cfg)]
    sub = _top_disagreeing(rows, "gradcam", k)
    ncol = len(methods) + 1
    fig, axes = plt.subplots(len(sub), ncol, figsize=(1.55 * ncol, 1.75 * len(sub)))
    axes = np.atleast_2d(axes)
    for i, r in enumerate(sub):
        s = samples[r["id"]]
        img = load_image(s).resize((224, 224))
        panels = [("humans", load_human_heatmap(cfg, s, human_source))]
        panels += [(m, model_map(cfg, s, m, (224, 224))) for m in methods]
        for j, (title, hm) in enumerate(panels):
            ax = axes[i, j]
            ax.imshow(img if hm is None else overlay(img, hm))
            ax.axis("off")
            if i == 0:
                ax.set_title(title, fontsize=7, color=INK)
    fig.tight_layout(pad=0.3)
    cfg.paths.figures_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.paths.figures_dir / "ablation_gallery.png"
    fig.savefig(out, dpi=130)
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

    def sort_key(e: dict) -> float:
        ms = e["methods"]
        return -(ms["gradcam"]["disagreement"] if "gradcam" in ms
                 else max(m["disagreement"] for m in ms.values()))

    payload = {
        "human_source": human_source,
        "methods": {"main": [m for m in main_methods(cfg)],
                    "ablation": [m for m in ablation_methods(cfg) if m not in cfg.model.methods]},
        "samples": sorted(by_id.values(), key=sort_key),
    }
    out = cfg.paths.results_dir / "agreement.json"
    with out.open("w") as fh:
        json.dump(payload, fh, indent=1)
    return out
