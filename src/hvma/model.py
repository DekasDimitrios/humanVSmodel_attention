"""CLIP saliency: where does the vision tower look when scored against an "artefact" prompt?

Two methods, both operating on Hugging Face `CLIPModel`:

- **gradcam**: Grad-CAM on the patch tokens entering the last vision transformer block (the
  output of its `layer_norm1`, i.e. jacobgil's `blocks[-1].norm1`). Not the block's *output*:
  the image embedding reads only the CLS token there, so patch tokens get exactly zero gradient.
  Channel weights are the mean gradient of the target score over the patch grid; the CAM is
  ReLU(sum_c w_c * A_c). This is the ViT adaptation popularised by jacobgil/pytorch-grad-cam,
  implemented here directly to avoid the dependency.
- **rollout**: attention rollout (Abnar & Zuidema 2020) from the CLS token to patches, with
  per-layer low-attention discarding as in jacobgil's implementation. Rollout is *not*
  prompt-conditioned; it shows what the image encoder attends to regardless of the question.
  Including it is deliberate — it is the baseline that tells you whether the prompt is doing
  anything.

Target score for gradcam: mean cosine similarity to `positive_prompts` minus mean similarity to
`negative_prompts`, i.e. "how much more artefact-like than clean is this image".

Output of both: float32 (grid, grid) maps in [0, 1] — 14x14 for patch16 at 224px. `report`/
`evaluate` upsample them to image size.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from .config import ModelCfg
from .heatmaps import normalize


def pick_device(pref: str) -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _embeds(out: torch.Tensor | object) -> torch.Tensor:
    """Projected embeddings from `get_{text,image}_features`.

    transformers 4.x returns the tensor; 5.x returns a model output with it in `pooler_output`.
    """
    return out if isinstance(out, torch.Tensor) else out.pooler_output  # type: ignore[attr-defined]


@dataclass
class SaliencyOutput:
    gradcam: np.ndarray | None
    rollout: np.ndarray | None
    score: float  # the positive-minus-negative similarity; useful as a per-image "artefactness"


class ClipSaliency:
    def __init__(self, cfg: ModelCfg):
        from transformers import CLIPModel, CLIPProcessor  # lazy import

        self.cfg = cfg
        self.device = pick_device(cfg.device)
        # eager attention is required for `output_attentions=True` (SDPA returns None)
        self.model = (
            CLIPModel.from_pretrained(cfg.name, attn_implementation="eager")
            .to(self.device)
            .eval()
        )
        self.processor = CLIPProcessor.from_pretrained(cfg.name)
        with torch.no_grad():
            self.pos = self._embed_text(cfg.positive_prompts)
            self.neg = self._embed_text(cfg.negative_prompts)
        vcfg = self.model.config.vision_config
        self.grid = vcfg.image_size // vcfg.patch_size

    # ------------------------------------------------------------------ helpers

    def _embed_text(self, prompts: list[str]) -> torch.Tensor | None:
        if not prompts:
            return None
        tok = self.processor(text=prompts, return_tensors="pt", padding=True).to(self.device)
        emb = _embeds(self.model.get_text_features(**tok))
        return torch.nn.functional.normalize(emb, dim=-1)

    def _pixel_values(self, img: Image.Image) -> torch.Tensor:
        return self.processor(images=img, return_tensors="pt")["pixel_values"].to(self.device)

    def _target(self, image_emb: torch.Tensor) -> torch.Tensor:
        """Positive-minus-negative similarity. Shape: scalar."""
        image_emb = torch.nn.functional.normalize(image_emb, dim=-1)
        score = torch.zeros((), device=self.device)
        if self.pos is not None:
            score = score + (image_emb @ self.pos.T).mean()
        if self.neg is not None:
            score = score - (image_emb @ self.neg.T).mean()
        return score

    # ------------------------------------------------------------------ methods

    def gradcam(self, img: Image.Image) -> tuple[np.ndarray, float]:
        """Grad-CAM over the patch tokens entering the last vision layer. Returns (map, score)."""
        pv = self._pixel_values(img)
        vision = self.model.vision_model
        captured: dict[str, torch.Tensor] = {}

        def hook(_m, _i, out):
            hs = out[0] if isinstance(out, tuple) else out
            hs.retain_grad()
            captured["hs"] = hs

        handle = vision.encoder.layers[-1].layer_norm1.register_forward_hook(hook)
        try:
            self.model.zero_grad(set_to_none=True)
            img_emb = _embeds(self.model.get_image_features(pixel_values=pv))
            score = self._target(img_emb)
            score.backward()
        finally:
            handle.remove()

        hs = captured["hs"]  # (1, 1+N, D)
        grads = hs.grad
        acts = hs[:, 1:, :].detach()  # drop CLS
        grads = grads[:, 1:, :]
        g = self.grid
        acts = acts.reshape(1, g, g, -1).permute(0, 3, 1, 2)  # (1, D, g, g)
        grads = grads.reshape(1, g, g, -1).permute(0, 3, 1, 2)
        weights = grads.mean(dim=(2, 3), keepdim=True)  # (1, D, 1, 1)
        cam = torch.relu((weights * acts).sum(dim=1))[0]  # (g, g)
        return normalize(cam.cpu().numpy()), float(score.item())

    @torch.no_grad()
    def rollout(self, img: Image.Image) -> np.ndarray:
        """Attention rollout CLS -> patches across all vision layers."""
        pv = self._pixel_values(img)
        out = self.model.vision_model(pixel_values=pv, output_attentions=True)
        attns = out.attentions  # tuple of (1, heads, T, T)
        T = attns[0].shape[-1]
        result = torch.eye(T, device=self.device)
        for a in attns:
            a = a[0].mean(dim=0)  # average heads -> (T, T)
            if self.cfg.rollout_discard_ratio > 0:
                flat = a.flatten()
                k = int(flat.numel() * self.cfg.rollout_discard_ratio)
                if k > 0:
                    idx = flat.topk(k, largest=False).indices
                    # never discard attention *to* the CLS column
                    idx = idx[idx % T != 0]
                    flat[idx] = 0
                    a = flat.view(T, T)
            a = a + torch.eye(T, device=self.device)  # residual connection
            a = a / a.sum(dim=-1, keepdim=True)
            result = a @ result
        cls_to_patches = result[0, 1:]  # (N,)
        g = self.grid
        return normalize(cls_to_patches.reshape(g, g).cpu().numpy())

    def __call__(self, img: Image.Image) -> SaliencyOutput:
        cam, score = (None, float("nan"))
        if "gradcam" in self.cfg.methods:
            cam, score = self.gradcam(img)
        roll = self.rollout(img) if "rollout" in self.cfg.methods else None
        if cam is None and roll is not None:
            # still report the artefactness score when gradcam is off
            with torch.no_grad():
                score = float(self._target(_embeds(self.model.get_image_features(
                    pixel_values=self._pixel_values(img)))).item())
        return SaliencyOutput(gradcam=cam, rollout=roll, score=score)
