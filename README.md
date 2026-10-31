# Do humans look where the model looks?

Comparing **where people tap when asked to find glitches in AI-generated images** (via
[Rapidata](https://rapidata.ai)'s crowd) against **where a CLIP model's saliency lands** when asked
the same question. The disagreements are the interesting part.

> Status: complete. Offline study on 40 images from Rapidata's public dataset, plus a small live
> Locate job (12 images, 144 responses) run on the account's free credit.

## Idea in one paragraph

Rapidata's Locate job asks real people to tap on visual artefacts in an image and returns the click
positions. Aggregating those clicks gives a human attention map. On the model side, Grad-CAM and
attention rollout on a CLIP vision tower, targeted at an "artefacts / glitches" text prompt, give a
model attention map. We score agreement per image (pointing game, IoU at the top-k% region,
correlation, NSS) and look at the tail: images where the crowd converges on a region the model
ignores, and vice versa.

## Data

Rapidata already published human artefact heatmaps in
[`Rapidata/text-2-image-Rich-Human-Feedback`](https://huggingface.co/datasets/Rapidata/text-2-image-Rich-Human-Feedback)
(`coherence_heatmap` column): crowd taps collected with the same kind of Locate task. All results
use that dataset, so the offline study costs nothing to reproduce.

To check that result against a fresh crowd, 12 of the 40 images also went through a live Locate
job via the Rapidata SDK: same question, 12 responses per image, 197 taps in total, $0.58 paid
from the account's free prepaid credit. Rapidata held it for a short manual review
(`ContentFlagged`) before it ran. The raw responses aren't published (they include labeller
demographics); see *Live check* below for the results.

## Setup

```bash
conda create -n hvma python=3.12 pip && conda activate hvma
# Pick the torch build for your GPU first. cu126 still ships Pascal (GTX 10xx) kernels;
# newer CUDA builds dropped them. CPU-only also works (40 images take a few minutes).
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[dev]"
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

Live job — **costs money** (12 images ≈ $0.58). Needs a Rapidata account: log in once with
`python -m rapidata login`, or put `RAPIDATA_CLIENT_ID` / `RAPIDATA_CLIENT_SECRET` in `.env`:

```bash
hvma estimate --limit 12           # prints the cost; charges nothing
hvma run --yes --limit 12          # launches the job, records it in results/live_job.json
hvma collect-results --wait        # waits for the crowd -> results/raw_locate.json
hvma parse-results results/raw_locate.json
hvma evaluate --human live && hvma report --human live   # writes *_live.* outputs
```

## Findings

40 images from the dataset, compared against the crowd's `coherence_heatmap`
(`results/summary.csv`, `results/ablation.csv`):

- **People tap on the subject, near the centre.** A model-free centre gaussian already reaches
  cc 0.50 and a pointing-game hit rate of 0.68. That is the bar any saliency method has to clear,
  and on this data none does.
- **Last-layer CLIP saliency is anti-correlated with people** (Grad-CAM cc −0.19, rollout −0.17;
  pointing 0.03, where a random map scores 0.10). 65% of its mass sits in the outer border of
  the image, versus 16% for the crowd, and it sits on the **flattest, least-textured** patches
  (texture ρ −0.37 for Grad-CAM, −0.55 for rollout; +0.41 for the crowd). The bias belongs to
  the **last block**: one block earlier, Grad-CAM moves onto the subject (cc +0.20, pointing
  0.42). Masking high-norm "artefact" tokens does not change it.
- **Registers fix attention in DINOv2** (cc 0.06 → 0.38, pointing 0.03 → 0.62). That map isn't
  prompt-conditioned, though: it finds the main object, not the glitch, and it still trails the
  centre prior. On these images, no CLIP or DINOv2 map localises glitches the way people do.

Caveats: 40 images (12 in the live check); the dataset gives aggregated heatmaps rather than raw
taps, so NSS is only reported for the live check; one prompt pair and one CLIP model (ViT-B/16).

### Live check: a fresh crowd on 12 images

The same question went to a fresh Rapidata crowd (12 responses per image), scored the same way
(`results/summary_live.csv`, `results/ablation_live.csv`). Since these images also have the
dataset's heatmap, the dataset crowd can be scored as a "method" predicting the fresh crowd's taps:
how well one group of people predicts another, a practical ceiling for any model.

| map | cc | pointing | NSS |
|---|---|---|---|
| dataset crowd (ceiling) | **0.59** | **1.00** | **1.76** |
| `dinov2reg_attn` | 0.29 | 0.67 | 0.77 |
| `center` | 0.29 | 0.33 | 0.70 |
| `gradcam_l10` | 0.18 | 0.58 | 0.47 |
| `gradcam` | −0.11 | 0.25 | −0.24 |
| `rollout` | −0.15 | 0.08 | −0.36 |

- **The human signal replicates.** Two independent crowds agree (cc 0.59), and the dataset
  crowd's peak falls in the fresh crowd's top region on all 12 images.
- **So do the model results.** Last-layer Grad-CAM and rollout are again anti-correlated with
  people and sit on flat, peripheral patches (border mass 0.62–0.63); `gradcam_l10` and the
  register map are again positive.
- **The best maps reach about half of the crowd ceiling,** and roughly tie the centre prior.
  With 12 images these are indicative, not precise.

### Where humans and the model disagree most

Using Grad-CAM one block before the last (`gradcam_l10`), the variant that escapes the edge bias,
the six least-agreeing images show a real disagreement. Both look at the subject, but at
different parts of it: **people flag a specific, local error** (a second cup, mismatched irises,
a deformed ferry) and Grad-CAM lands elsewhere (hair, sky, highlights).

That looked like Grad-CAM preferring texture-rich regions, so we measured it on all 40 images:
it doesn't. `gradcam_l10` follows image texture *as much as people do* (ρ 0.21 vs 0.24 once the
centre prior is partialled out; paired Wilcoxon p = 0.96), and in five of these six images the
crowd's taps track texture more than Grad-CAM does. Whatever pulls the model away from the
error, it isn't detail or edges. See *Texture* below.

![disagreement gallery, gradcam_l10](results/figures/disagreement_gallery_gradcam_l10.png)

1. `s0020` *Einstein drinking coffee* — people tap the two cups (one at his mouth, one floating
   below); Grad-CAM lights his hair and jacket and misses the duplicated cup.
2. `s0030` *dead city* — the crowd marks the fire-lit arches at lower left; Grad-CAM marks the
   sky and spires.
3. `s0036` *crystal tiger* — people tap the muzzle; Grad-CAM scatters over bokeh highlights and
   ear tips.
4. `s0004` *eyes* — people tap both irises (two mismatched eyes fused into one face); Grad-CAM
   marks the brow and a corner of skin.
5. `s0016` *cake shaped like New York* — people converge on the malformed ferry; Grad-CAM spreads
   over the skyline.
6. `s0029` *crystal screensaver* — the mildest case (cc −0.10): people tap the cluster's base and
   a stray rock, Grad-CAM the crystal tips. With a fully stylised subject, neither has an
   obvious error to find.

<details>
<summary>Same view for last-layer Grad-CAM: all edge bias</summary>

![disagreement gallery, gradcam](results/figures/disagreement_gallery.png)

All six are the edge bias at work: people tap on the subject while Grad-CAM tiles the empty
background.

1. `s0010` *banana character* — face and stubby hands vs. the grey backdrop.
2. `s0025` *anthropomorphic Thwomp woman* — the torso's anatomy vs. the white background.
3. `s0037` *woman with flowing hair* — the outstretched hand vs. the empty wall.
4. `s0031` *helicopter with car wheels* — where car and helicopter fuse vs. tarmac and sky.
5. `s0024` *chakat* — oversized ears and face vs. painted foliage.
6. `s0000` *wood-cut otter* — the tail in the whirlpool vs. the print's frame.

</details>

### Edge-bias ablation (dataset heatmaps, 40 images)

Last-layer CLIP Grad-CAM and rollout put ~65% of their mass in the outer border of the image
(a uniform map: 49%; human taps: 16%) and end up *anti*-correlated with people. A model-free
centre gaussian (`center`) beats both by a wide margin, so every number here should be read
against it. Each ablation variant tests one explanation (`results/ablation.csv`):

| map | what it tests | cc | pointing | border mass | texture ρ (│centre) |
|---|---|---|---|---|---|
| human taps | reference | — | — | 0.16 | +0.41 (+0.24) |
| `center` | centre-bias baseline | **0.50** | **0.68** | 0.12 | +0.37 (−0.02) |
| `gradcam` | main method, last block | −0.19 | 0.03 | 0.65 | −0.37 (−0.29) |
| `rollout` | main method, all blocks | −0.17 | 0.03 | 0.64 | −0.55 (−0.49) |
| `gradcam_l10` / `l8` / `l6` / `l4` | earlier blocks | 0.20 / 0.02 / 0.02 / 0.13 | 0.42 / 0.30 / 0.25 / 0.38 | 0.36–0.50 | +0.29 / 0.00 / −0.03 / +0.24 |
| `gradcam_masked` | in-paint high-norm tokens | −0.18 | 0.03 | 0.64 | −0.32 (−0.25) |
| `dinov2_attn` | DINOv2 CLS attention | 0.06 | 0.03 | 0.49 | +0.25 (+0.16) |
| `dinov2reg_attn` | same, model trained with registers | 0.38 | 0.62 | 0.35 | +0.43 (+0.33) |

- The edge bias comes from the **last block**: one block earlier (`gradcam_l10`) the map moves
  onto the subject and correlates positively.
- **Masking high-norm tokens does nothing.** CLIP has ~10 of them per image, but only 7% sit in
  the border and 3% of Grad-CAM's top patches are outliers, so they are not what Grad-CAM sees.
- **Registers fix DINOv2.** Plain DINOv2 attention is spread evenly (border mass 0.49); the register
  variant lands on the subject. It isn't prompt-conditioned, so this is object-centric
  attention rather than glitch detection, and it still trails `center`.

**Texture.** Per image, the Spearman correlation between a map and the image's texture energy
(mean Sobel gradient magnitude per ViT patch, on the 224 px input), averaged over the 40 images.
The value in brackets partials out the centre prior: textured regions tend to be the subject,
which tends to be central, and the centre map dropping from +0.37 to −0.02 shows the control
does its job. Paired Wilcoxon tests against the crowd on the same images are in
`results/ablation.csv` (`texture_rho_p`, `texture_rho_ctr_p`).

- Last-layer maps prefer the flattest patches, even after the centre control (p < 1e-10 vs the
  crowd). That fits the edge bias: borders and backgrounds are the flat parts of these images.
- `gradcam_l10` matches the crowd's texture preference (p = 0.96 with the control), so texture
  does not explain where it disagrees with people.

![edge-bias ablation](results/figures/ablation_edge_bias.png)

## Notes

- Built with Claude Code; `CLAUDE.md` holds the working agreement.
- Code: MIT (see `LICENSE`).
- Not affiliated with Rapidata. Dataset and the images in the gallery:
  [`Rapidata/text-2-image-Rich-Human-Feedback`](https://huggingface.co/datasets/Rapidata/text-2-image-Rich-Human-Feedback),
  © Rapidata, Apache-2.0.
