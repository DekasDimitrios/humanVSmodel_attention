# CLAUDE.md — Human vs. Model Attention on AI-Generated Images

## What this project is

A weekend-scale experiment: **do humans and a vision model look at the same places when asked
"what looks wrong in this AI-generated image?"**

- **Human side:** click/tap heatmaps from Rapidata's crowd. Two sources:
  1. *Offline (free):* the `coherence_heatmap` column of the Hugging Face dataset
     `Rapidata/text-2-image-Rich-Human-Feedback`. Use this to develop and validate everything.
  2. *Live (costs money):* a Rapidata **Locate** job via the `rapidata` Python SDK on a small set
     of fresh images. This is the "did something cool with the product" part of the job application.
- **Model side:** saliency maps from a CLIP vision tower — Grad-CAM on the last transformer block
  and attention rollout — targeted at an "artifact / glitch" text prompt, read against a
  model-free `center` baseline (human taps are strongly centre-biased). An edge-bias ablation
  (earlier-layer Grad-CAM, high-norm-token masking, DINOv2 with/without registers) is reported
  separately in `results/ablation.csv`.
- **Output:** per-image agreement metrics (pointing game, IoU@top-k, correlation, NSS), a CSV,
  figures, and a tiny FastAPI viewer that sorts images by disagreement so the interesting cases
  surface first.

The author is an ML/XAI person (Grad-CAM, attention maps, CLIP/DINOv2, React+FastAPI dashboards).
Favour clarity and correctness over cleverness; this repo will be read by hiring engineers.

## Repository layout

```
configs/default.yaml      all tunables (dataset, N images, model, sigma, thresholds, Rapidata job params)
src/hvma/                 the package ("human vs model attention")
  config.py               load YAML + .env into a Config dataclass
  data.py                 stream HF dataset, save images, convert dataset heatmaps to arrays
  heatmaps.py             points -> gaussian heatmap; normalise, resize, threshold
  rapidata_job.py         build/estimate/run a Locate job; parse raw results into point sets
  model.py                CLIP saliency: gradcam + attention rollout; ablation maps (earlier-layer
                          gradcam, high-norm-token masking, DINOv2 with/without registers)
  metrics.py              agreement metrics between two heatmaps
  report.py               overlays, CSV, summary figures, agreement.json for the viewer;
                          main methods -> summary.csv, edge-bias ablation -> ablation.csv
  cli.py                  typer CLI; `hvma --help`
viewer/                   FastAPI app serving results/ + a single static index.html
tests/                    pytest unit tests for heatmaps + metrics (pure numpy, fast)
data/                     local images + manifest (gitignored)
results/                  raw job JSON, saliency .npy, overlays, CSV (gitignored except README)
```

## Commands

```bash
conda activate hvma                 # env setup: see README → Setup (torch from the cu126 index)
pip install -e ".[dev]"            # install
pytest -q                           # unit tests (no network, no GPU)
hvma fetch --n 40                   # stream N samples from HF -> data/images + data/manifest.json
hvma inspect-dataset                # print the raw shape/type of coherence_heatmap for one sample
hvma saliency                       # compute model saliency for every image in the manifest
hvma evaluate --human dataset       # metrics vs. the dataset heatmaps (free path)
hvma estimate                       # cost estimate for the live Locate job; NEVER runs it
hvma run --yes                      # launch the live Locate job (asks for --yes; prints estimate first)
hvma parse-results results/raw_locate.json   # raw SDK JSON -> point sets -> human heatmaps
hvma evaluate --human live          # metrics vs. live crowd heatmaps
hvma report                         # overlays + CSV + figures + results/agreement.json
hvma serve                          # viewer at http://127.0.0.1:8000
```

## Hard rules

1. **Never launch a paid Rapidata job without the `--yes` flag and printing `estimated_cost` first.**
   `hvma run` must refuse if the estimate exceeds `rapidata.max_cost` in the config.
2. **Cap live-job size.** `rapidata.max_datapoints` in config (default 40) is a hard ceiling.
3. **Secrets only via `.env`** (`RAPIDATA_CLIENT_ID`, `RAPIDATA_CLIENT_SECRET`). Never commit them,
   never print them. Authentication details: https://docs.rapidata.ai/authentication/
4. **Verify SDK surface before trusting it.** Parts of the Rapidata result schema for Locate jobs
   were not documented when this skeleton was written. Before relying on `rapidata_job.py`:
   - run `/verify-sdk` (see `.claude/commands/verify-sdk.md`), or
   - `python -c "import rapidata, inspect; help(rapidata.RapidataClient)"` and inspect
     `create_locate_job_definition`, `assign_job`, `get_results`, `estimated_cost`.
   - After the first live run, `hvma parse-results --dump-schema` and fix `extract_points()` if the
     heuristic parser mis-reads the structure.
5. **Keep the free path working.** Everything except `estimate`/`run` must work offline once
   `hvma fetch` has run, with no Rapidata credentials.
6. Do not add heavy dependencies for small wins. Current stack: torch, transformers, datasets,
   numpy, scipy, Pillow, matplotlib, typer, pyyaml, fastapi, uvicorn.

## Known uncertainties (resolve these first when picking the project up)

- [x] Exact shape/encoding of `coherence_heatmap` in the HF dataset. **Resolved:** 2-D list of
      float16 at image resolution (images are 1024x1024), dense, max ~0.16 before normalising;
      ~30% of rows have none (`dataset.require_heatmap` skips them). Handled as-is by
      `data.dataset_heatmap_to_array`; see `tests/test_data.py`.
- [x] Whether `create_locate_job_definition(datapoints=...)` accepts local file paths.
      **Resolved (SDK 3.27 source + bundled guide `python -m rapidata skill`):** yes, local paths
      are uploaded. Still confirm with ONE datapoint via `hvma estimate --limit 1` (no charge).
- [ ] Field names in Locate results (points per response, normalised vs. pixel coordinates).
      SDK 3.27 confirms the envelope (`{"results": [{..., "detailedResults": [...],
      "privateMetadata": {...}}]}`) but not the per-point shape, which comes from the server.
      `extract_points` handles both coordinate conventions; check with `--dump-schema`.
- [x] How result entries are keyed back to the input datapoints. **Resolved:** each datapoint
      is submitted with `private_metadata={"id": <sample id>}`, returned on every result entry.

## Conventions

- Heatmaps are `float32` numpy arrays in `[0, 1]`, shape `(H, W)`, same size as the image unless a
  function says otherwise. Normalise with `heatmaps.normalize` (max-norm) before comparing.
- Points are `(x, y)` tuples in **normalised** image coordinates (0–1, origin top-left).
- Every artefact written to `results/` is keyed by the sample `id` from `data/manifest.json`.
- Type hints everywhere; small pure functions; docstrings say *why* when it isn't obvious.
- Tests for anything numerical (`metrics.py`, `heatmaps.py`). Model/network code is exercised by
  the CLI, not unit tests.

## Definition of done (for the application)

- `hvma evaluate --human dataset` runs on ≥30 images and `results/summary.csv` exists.
- At least one small live Locate job ran (≥10 images, ~10–15 responses each) and its results were
  evaluated with `--human live`.
- `results/figures/disagreement_gallery.png` shows the 6 images where humans and the model disagree
  most, with a one-line interpretation each in the README.
- README has: 1 paragraph on the idea, 1 figure, 3 bullet findings, how to reproduce in 5 commands.
- Mention in the README that the repo was built with Claude Code (one sentence; no more).
