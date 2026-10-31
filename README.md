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

## Notes

- Built with Claude Code; `CLAUDE.md` holds the working agreement.
- Not affiliated with Rapidata. Dataset © Rapidata, used under its Hugging Face licence.
