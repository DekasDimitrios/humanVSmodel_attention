"""Configuration: YAML tunables + .env secrets, exposed as a small typed object.

Kept deliberately simple (no pydantic) so a reader can see every knob in one place.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

DEFAULT_CONFIG = Path("configs/default.yaml")


@dataclass
class PathsCfg:
    data_dir: Path = Path("data")
    results_dir: Path = Path("results")

    @property
    def images_dir(self) -> Path:
        return self.data_dir / "images"

    @property
    def manifest(self) -> Path:
        return self.data_dir / "manifest.json"

    @property
    def human_dir(self) -> Path:
        """Human heatmaps as .npy, in subfolders `dataset/` and `live/`."""
        return self.results_dir / "human"

    @property
    def saliency_dir(self) -> Path:
        return self.results_dir / "saliency"

    @property
    def figures_dir(self) -> Path:
        return self.results_dir / "figures"

    @property
    def overlays_dir(self) -> Path:
        return self.results_dir / "overlays"


@dataclass
class DatasetCfg:
    name: str = "Rapidata/text-2-image-Rich-Human-Feedback"
    split: str = "train"
    n_samples: int = 40
    seed: int = 7
    keep_prob: float = 0.25
    require_heatmap: bool = True
    heatmap_column: str = "coherence_heatmap"
    max_coherence_score: float | None = None


@dataclass
class HeatmapCfg:
    sigma_frac: float = 0.04
    topk_fracs: list[float] = field(default_factory=lambda: [0.1, 0.2, 0.3])


@dataclass
class ModelCfg:
    name: str = "openai/clip-vit-base-patch16"
    device: str = "auto"
    methods: list[str] = field(default_factory=lambda: ["gradcam", "rollout", "center"])
    positive_prompts: list[str] = field(default_factory=list)
    negative_prompts: list[str] = field(default_factory=list)
    rollout_discard_ratio: float = 0.9
    center_sigma_frac: float = 0.2


@dataclass
class AblationCfg:
    enabled: bool = True
    gradcam_layers: list[int] = field(default_factory=lambda: [4, 6, 8, 10])
    outlier_norm_ratio: float = 2.0
    register_models: dict[str, str] = field(default_factory=lambda: {
        "dinov2": "facebook/dinov2-base",
        "dinov2reg": "facebook/dinov2-with-registers-base",
    })

    def methods(self) -> list[str]:
        """Names of the ablation maps, in the order they are reported."""
        if not self.enabled:
            return []
        return ([f"gradcam_l{k}" for k in self.gradcam_layers]
                + ["gradcam_masked", "rollout_masked"]
                + [f"{name}_attn" for name in self.register_models])


@dataclass
class RapidataCfg:
    audience_id: str = "global"
    job_name: str = "Human vs model attention"
    instruction: str = "Tap on any part of the image that looks wrong, glitchy or artificial."
    responses_per_datapoint: int = 12
    max_datapoints: int = 40
    max_cost: float = 15.0
    raw_results_file: str = "raw_locate.json"


@dataclass
class Config:
    paths: PathsCfg = field(default_factory=PathsCfg)
    dataset: DatasetCfg = field(default_factory=DatasetCfg)
    heatmap: HeatmapCfg = field(default_factory=HeatmapCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    ablation: AblationCfg = field(default_factory=AblationCfg)
    rapidata: RapidataCfg = field(default_factory=RapidataCfg)


def _build(cls: type, raw: dict[str, Any] | None) -> Any:
    raw = raw or {}
    fields = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
    unknown = set(raw) - fields
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    kwargs = dict(raw)
    if cls is PathsCfg:
        kwargs = {k: Path(v) for k, v in kwargs.items()}
    return cls(**kwargs)


def load_config(path: Path | str = DEFAULT_CONFIG) -> Config:
    """Load YAML config and `.env`. Env vars are not stored on the Config; the SDK reads them."""
    load_dotenv()  # harmless if no .env exists
    path = Path(path)
    raw: dict[str, Any] = {}
    if path.exists():
        with path.open() as fh:
            raw = yaml.safe_load(fh) or {}
    return Config(
        paths=_build(PathsCfg, raw.get("paths")),
        dataset=_build(DatasetCfg, raw.get("dataset")),
        heatmap=_build(HeatmapCfg, raw.get("heatmap")),
        model=_build(ModelCfg, raw.get("model")),
        ablation=_build(AblationCfg, raw.get("ablation")),
        rapidata=_build(RapidataCfg, raw.get("rapidata")),
    )


def have_rapidata_credentials() -> bool:
    return bool(os.getenv("RAPIDATA_CLIENT_ID") and os.getenv("RAPIDATA_CLIENT_SECRET"))
