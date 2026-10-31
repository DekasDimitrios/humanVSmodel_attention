Run the whole offline pipeline end to end and fix whatever breaks. No Rapidata credentials needed.

1. `pip install -e ".[dev]"` if `hvma` is not on PATH.
2. `pytest -q` — must be green before anything else.
3. `hvma inspect-dataset` — read the output and confirm `coherence_heatmap`'s shape/encoding.
   If `data.dataset_heatmap_to_array` does not handle it, fix the function, add a unit test with a
   tiny synthetic example of the real encoding, and tick the box in CLAUDE.md.
4. `hvma fetch --n 8` (small first), then `hvma saliency`, `hvma evaluate --human dataset`,
   `hvma report`.
5. Open `results/figures/disagreement_gallery.png` and `results/overlays/` and sanity-check:
   - human overlays should look like click clusters, not noise or a uniform wash
   - gradcam maps should vary between images (if every map is identical, the target or hook is wrong)
   - rollout maps should differ from gradcam
6. If all good, re-run `hvma fetch --n 40` and the rest.
7. Report: the summary table from `hvma evaluate`, and one sentence on whether the maps look sane.
