"""CLIP saliency: where does the vision tower look when scored against an "artefact" prompt?

Two main methods, both operating on Hugging Face `CLIPModel`:

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

Edge-bias ablation (see `AblationCfg`; reported separately from the main methods):

- **gradcam_l<k>**: the same Grad-CAM taken at the input of an earlier block k. All layers come
  from one forward/backward pass.
- **gradcam_masked / rollout_masked**: the main maps with high-norm "artefact" tokens
  (`heatmaps.outlier_tokens`, measured on the input to the last block) in-painted from their
  neighbours.
- **<name>_attn** (`ClsAttention`): last-layer CLS->patch attention of DINOv2 without and with
  registers — a ViT trained to remove those artefact tokens. Not prompt-conditioned, so it
  tests whether the edge bias is generic ViT behaviour rather than something about CLIP.

Target score for gradcam: mean cosine similarity to `positive_prompts` minus mean similarity to
`negative_prompts`, i.e. "how much more artefact-like than clean is this image".

All outputs: float32 (grid, grid) maps in [0, 1] — 14x14 for CLIP patch16 at 224px, 16x16 for
DINOv2 patch14 at 224px. `report`/`evaluate` upsample them to image size.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image

from .config import AblationCfg, ModelCfg
from .heatmaps import inpaint_tokens, normalize, outlier_tokens


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
    maps: dict[str, np.ndarray] = field(default_factory=dict)  # method name -> (grid, grid) map
    score: float = float("nan")  # positive-minus-negative similarity: per-image "artefactness"
    n_outliers: int = 0  # high-norm tokens found at the input of the last block


class ClipSaliency:
    def __init__(self, cfg: ModelCfg, ablation: AblationCfg | None = None):
        from transformers import CLIPModel, CLIPProcessor  # lazy import

        self.cfg = cfg
        self.ablation = ablation if ablation is not None and ablation.enabled else None
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
        self.last = vcfg.num_hidden_layers - 1

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

    def _to_grid(self, tokens: torch.Tensor) -> torch.Tensor:
        """(1, 1+N, D) hidden states -> (1, D, g, g) patch grid (CLS dropped)."""
        g = self.grid
        return tokens[:, 1:, :].reshape(1, g, g, -1).permute(0, 3, 1, 2)

    # ------------------------------------------------------------------ methods

    def gradcam(
        self, img: Image.Image, layers: list[int] | None = None
    ) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], float]:
        """Grad-CAM over the patch tokens entering each block in `layers` (default: the last).

        One forward/backward pass serves every layer. Returns (cams, token_norms, score), where
        cams/token_norms map block index -> (grid, grid) array; the norms are those of the
        block's *pre*-LayerNorm input, where high-norm outlier tokens are visible.
        """
        layers = sorted(set(layers or [self.last]))
        pv = self._pixel_values(img)
        blocks = self.model.vision_model.encoder.layers
        captured: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

        def make_hook(k: int):
            def hook(_m, inputs, out):
                out.retain_grad()
                captured[k] = (inputs[0].detach(), out)
            return hook

        handles = [blocks[k].layer_norm1.register_forward_hook(make_hook(k)) for k in layers]
        try:
            self.model.zero_grad(set_to_none=True)
            img_emb = _embeds(self.model.get_image_features(pixel_values=pv))
            score = self._target(img_emb)
            score.backward()
        finally:
            for h in handles:
                h.remove()

        cams: dict[int, np.ndarray] = {}
        norms: dict[int, np.ndarray] = {}
        for k in layers:
            pre, ln_out = captured[k]
            acts = self._to_grid(ln_out.detach())  # (1, D, g, g)
            grads = self._to_grid(ln_out.grad)
            weights = grads.mean(dim=(2, 3), keepdim=True)  # (1, D, 1, 1)
            cam = torch.relu((weights * acts).sum(dim=1))[0]  # (g, g)
            cams[k] = normalize(cam.cpu().numpy())
            norms[k] = self._to_grid(pre).norm(dim=1)[0].cpu().numpy()
        return cams, norms, float(score.item())

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
        """Every CLIP map the config asks for (main methods + CLIP-side ablations)."""
        abl = self.ablation
        layers = [self.last] + (list(abl.gradcam_layers) if abl else [])
        cams, norms, score = self.gradcam(img, layers)
        out = SaliencyOutput(score=score)
        if "gradcam" in self.cfg.methods or abl:
            out.maps["gradcam"] = cams[self.last]
        if "rollout" in self.cfg.methods or abl:
            out.maps["rollout"] = self.rollout(img)
        if abl:
            for k in abl.gradcam_layers:
                out.maps[f"gradcam_l{k}"] = cams[k]
            mask = outlier_tokens(norms[self.last], abl.outlier_norm_ratio)
            out.n_outliers = int(mask.sum())
            out.maps["gradcam_masked"] = inpaint_tokens(out.maps["gradcam"], mask)
            out.maps["rollout_masked"] = inpaint_tokens(out.maps["rollout"], mask)
        return out


class ClsAttention:
    """Last-layer CLS -> patch attention (mean over heads) of a DINOv2-family model.

    Register tokens, if the model has them, sit between CLS and the patches and are dropped.
    """

    def __init__(self, model_id: str, device: torch.device):
        from transformers import AutoImageProcessor, AutoModel  # lazy import

        self.device = device
        self.model = (
            AutoModel.from_pretrained(model_id, attn_implementation="eager").to(device).eval()
        )
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.n_registers = int(getattr(self.model.config, "num_register_tokens", 0) or 0)

    @torch.no_grad()
    def __call__(self, img: Image.Image) -> np.ndarray:
        pv = self.processor(images=img, return_tensors="pt")["pixel_values"].to(self.device)
        attn = self.model(pixel_values=pv, output_attentions=True).attentions[-1]  # (1, H, T, T)
        cls = attn[0].mean(dim=0)[0, 1 + self.n_registers:]
        g = int(round(cls.numel() ** 0.5))
        return normalize(cls.reshape(g, g).cpu().numpy())
