# Do humans look where the model looks?

Comparing **where people tap when asked to find glitches in AI-generated images** (via
[Rapidata](https://rapidata.ai)'s crowd) against **where a CLIP model's saliency lands** when asked
the same question. The disagreements are the interesting part.

> Status: skeleton. See `CLAUDE.md` for the plan, conventions and the open questions to resolve
> first.

## Idea in one paragraph

Rapidata's Locate job asks real people to tap on visual artefacts in an image and returns the click
positions. Aggregating those clicks gives a human attention map. On the model side, Grad-CAM and
attention rollout on a CLIP vision tower, targeted at an "artefacts / glitches" text prompt, give a
model attention map. We score agreement per image (pointing game, IoU at the top-k% region,
correlation, NSS) and look at the tail: images where the crowd converges on a region the model
ignores, and vice versa.

## Two phases

1. **Offline (free).** Rapidata already published human artefact heatmaps in
   [`Rapidata/text-2-image-Rich-Human-Feedback`](https://huggingface.co/datasets/Rapidata/text-2-image-Rich-Human-Feedback)
   (`coherence_heatmap` column). The whole pipeline is developed and validated against those.
2. **Live (small, paid).** A Locate job on a handful of fresh images through the Python SDK, to
   check the pipeline end-to-end on real crowd output and to use the product for real.

## Setup

```bash
conda create -n hvma python=3.12 pip && conda activate hvma
# Pick the torch build for your GPU first. cu126 still ships Pascal (GTX 10xx) kernels;
# newer CUDA builds dropped them. CPU-only also works (40 images take a few minutes).
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[dev]"
cp .env.example .env       # only needed for the live Rapidata job
pytest -q
```

Run every command from the repo root (config and data paths are relative). Fetching streams
the dataset and peaks at ~2.5 GB RAM.

## Reproduce

```bash
hvma fetch --n 40          # stream 40 samples from Hugging Face
hvma saliency              # CLIP Grad-CAM + attention rollout
hvma evaluate --human dataset
hvma report && hvma serve  # open http://127.0.0.1:8000
```

Live job (needs `RAPIDATA_CLIENT_ID` / `RAPIDATA_CLIENT_SECRET` in `.env`):

```bash
hvma estimate              # prints the cost; charges nothing
hvma run --yes             # launches the Locate job and saves results/raw_locate.json
hvma parse-results results/raw_locate.json
hvma evaluate --human live
```

## Findings

_TODO — three bullets and one figure, once the numbers are in._

### Edge-bias ablation (dataset heatmaps, 40 images)

Last-layer CLIP Grad-CAM and rollout put ~65% of their mass in the outer border of the image
(a uniform map: 49%; human taps: 16%) and end up *anti*-correlated with people. A model-free
centre gaussian (`center`) beats both by a wide margin, so every number here should be read
against it. Each ablation variant tests one explanation (`results/ablation.csv`):

| map | what it tests | cc | pointing | border mass |
|---|---|---|---|---|
| `center` | centre-bias baseline | **0.50** | **0.68** | 0.12 |
| `gradcam` | main method, last block | −0.19 | 0.03 | 0.65 |
| `gradcam_l10` / `l8` / `l6` / `l4` | earlier blocks | 0.20 / 0.02 / 0.02 / 0.13 | 0.42 / 0.30 / 0.25 / 0.38 | 0.36–0.50 |
| `gradcam_masked` | in-paint high-norm tokens | −0.18 | 0.03 | 0.64 |
| `dinov2_attn` | DINOv2 CLS attention | 0.06 | 0.03 | 0.49 |
| `dinov2reg_attn` | same, model trained with registers | 0.38 | 0.62 | 0.35 |

- The edge bias comes from the **last block**: one block earlier (`gradcam_l10`) the map moves
  onto the subject and correlates positively.
- **Masking high-norm tokens does nothing.** CLIP has ~10 of them per image, but only 7% sit in
  the border and 3% of Grad-CAM's top patches are outliers, so they are not what Grad-CAM sees.
- **Registers fix DINOv2.** Plain DINOv2 attention is spread evenly (border mass 0.49); the register
  variant lands on the subject. It isn't prompt-conditioned, so this is object-centric
  attention rather than glitch detection, and it still trails `center`.

![edge-bias ablation](results/figures/ablation_edge_bias.png)

## Notes

- Built with Claude Code; `CLAUDE.md` holds the working agreement.
- Not affiliated with Rapidata. Dataset © Rapidata, used under its Hugging Face licence.
