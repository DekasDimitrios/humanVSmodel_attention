Prepare and (only with my explicit go-ahead) run the small live Rapidata Locate job.

Preconditions — stop and tell me if any fails:
- `/free-path` has been completed and `results/metrics_dataset.csv` exists.
- `.env` has `RAPIDATA_CLIENT_ID` and `RAPIDATA_CLIENT_SECRET` (do not print them).
- `/verify-sdk` has been run in this repo.

Steps:
1. `hvma estimate --limit 1` — confirms the SDK accepts our datapoint refs (local paths or URLs).
   If it fails on local paths, stop and tell me; we will host the images and set
   `live_datapoint_ref` in the manifest instead.
2. `hvma estimate --limit 12` — show me the printed estimate.
3. **Wait for me to type "go".** Then `hvma run --yes --limit 12`.
4. `hvma parse-results results/raw_locate.json --dump-schema` — paste the type tree here.
5. `hvma parse-results results/raw_locate.json`. If points-per-sample is 0, fix `extract_points`
   for the real schema, add a test with a redacted real entry, and re-run.
6. `hvma evaluate --human live` and `hvma report --human live`.
7. Compare the live vs dataset summary tables and tell me in 3 bullets what differs.
