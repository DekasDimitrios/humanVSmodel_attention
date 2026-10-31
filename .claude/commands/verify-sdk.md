Verify the installed `rapidata` SDK against what `src/hvma/rapidata_job.py` assumes, and fix any mismatch.

Steps:
1. `pip show rapidata` — note the version.
2. Introspect, without making any network call:
   ```
   python - <<'EOF'
   import inspect, rapidata
   from rapidata import RapidataClient
   print([n for n in dir(rapidata) if not n.startswith('_')])
   for name in ("create_locate_job_definition",):
       # job manager lives at client.job in the docs; find the class without instantiating a client
       for cls_name, cls in inspect.getmembers(rapidata, inspect.isclass):
           if hasattr(cls, name):
               print(cls_name, name, inspect.signature(getattr(cls, name)))
   EOF
   ```
   Also locate and print the signatures/docstrings of `assign_job`, `get_results`, `estimated_cost`,
   and whatever class represents a Locate *result* (grep the package source for "locate" / "Locate").
3. Compare with `rapidata_job.py`:
   - parameter names in `build_job_definition`
   - whether `datapoints` accepts local file paths (read the docstring / source; do not guess)
   - the result object type returned by `get_results` and how it serialises (`_to_jsonable`)
   - the shape of locate results, if the SDK source reveals it; update `extract_points` and the
     docstring at the top of the module accordingly.
4. Update the "Known uncertainties" checklist in `CLAUDE.md` for anything you resolved.
5. Run `pytest -q`. Report what changed in ≤5 bullet points.

Do not run `hvma run`. `hvma estimate` is allowed if credentials are present (it charges nothing).
