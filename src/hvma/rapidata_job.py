"""Rapidata Locate job: build, estimate, run, and parse.

Grounded in the public docs (https://docs.rapidata.ai/examples/locate_job/ and
/cost_estimates/). Two things were NOT documented and are handled defensively:

1. The per-response structure of Locate results. Resolved by the first live job: see
   `extract_taps` (taps are in percent of image size; `aggregatedResults` duplicates them).
2. How result entries map back to input datapoints. Each datapoint is submitted with
   `private_metadata={"id": <sample id>}`, which the SDK (3.27) returns on every result entry, so
   `match_results_to_manifest` finds the id among the entry's strings; it also tries `assetUrls`
   filenames and finally falls back to index order.

Guardrails (CLAUDE.md hard rules 0–2): `run_job` reports the estimate first, refuses above
`max_cost` or `max_datapoints`, and only launches with `confirm=True`.

A launched job is recorded in results/live_job.json the moment it is assigned, because it may
not finish in one sitting: Rapidata can hold a job for manual review (e.g. `ContentFlagged`)
for hours. `collect_results` picks it up again by id; it never creates or pays for a new job.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config
from .data import Sample, save_manifest
from .heatmaps import Point, points_to_heatmap


class RefusedError(RuntimeError):
    """Raised when a guardrail blocks a paid action."""


class JobNotReadyError(RuntimeError):
    """The job exists but has no final results yet (running, or held by Rapidata)."""

    def __init__(self, status: str, detail: str = ""):
        self.status = status
        super().__init__(f"job status {status}" + (f": {detail}" if detail else ""))


# States that never finish on their own (SDK 3.27): someone has to approve, top up or resume.
HELD_STATUSES = ("ManualApproval", "SpendLimited", "Paused", "Blocked")


# --------------------------------------------------------------------------- building & running


def _client():
    from rapidata import RapidataClient  # lazy: keeps the free path import-light

    return RapidataClient()  # reads RAPIDATA_CLIENT_ID / RAPIDATA_CLIENT_SECRET from env


def datapoint_refs(samples: list[Sample]) -> list[str]:
    """What we hand the SDK as `datapoints`.

    The SDK (3.27) accepts local paths and uploads them, as well as public URLs. To submit
    hosted copies instead, set `Sample.live_datapoint_ref` to the URL for each sample.
    """
    return [s.live_datapoint_ref or s.image_path for s in samples]


def build_job_definition(client: Any, cfg: Config, samples: list[Sample]) -> Any:
    rc = cfg.rapidata
    if len(samples) > rc.max_datapoints:
        raise RefusedError(
            f"{len(samples)} datapoints exceeds rapidata.max_datapoints={rc.max_datapoints}"
        )
    return client.job.create_locate_job_definition(
        name=rc.job_name,
        instruction=rc.instruction,
        datapoints=datapoint_refs(samples),
        responses_per_datapoint=rc.responses_per_datapoint,
        private_metadata=[{"id": s.id} for s in samples],  # echoed back on each result entry
    )


def _estimate_dict(jd: Any) -> dict[str, Any]:
    est = jd.estimated_cost  # blocks briefly while Rapidata prices the definition
    return {
        "estimated_cost": float(est.estimated_cost),
        "datapoint_count": int(est.datapoint_count),
        "required_responses": int(est.required_responses),
    }


def estimate(cfg: Config, samples: list[Sample]) -> dict[str, Any]:
    """Cost estimate without launching anything. Creates a job definition; charges nothing."""
    return _estimate_dict(build_job_definition(_client(), cfg, samples))


def job_record_path(cfg: Config) -> Path:
    return cfg.paths.results_dir / "live_job.json"


def load_job_record(cfg: Config) -> dict[str, Any]:
    p = job_record_path(cfg)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found — no live job has been launched from here")
    with p.open() as fh:
        return json.load(fh)


def _save_job_record(cfg: Config, record: dict[str, Any]) -> None:
    p = job_record_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as fh:
        json.dump(record, fh, indent=2)


def run_job(
    cfg: Config,
    samples: list[Sample],
    confirm: bool = False,
    on_estimate: Callable[[dict[str, Any]], None] | None = None,
) -> str:
    """Price the job, then (only with `confirm=True`) launch it and record it. Returns the job id.

    One job definition serves both the estimate and the launch. Does not wait for results:
    call `collect_results` (CLI: `hvma collect-results`) afterwards.
    """
    client = _client()
    jd = build_job_definition(client, cfg, samples)
    est = _estimate_dict(jd)
    if on_estimate is not None:
        on_estimate(est)
    if est["estimated_cost"] > cfg.rapidata.max_cost:
        raise RefusedError(f"estimated cost {est['estimated_cost']:.2f} exceeds "
                           f"rapidata.max_cost={cfg.rapidata.max_cost}")
    if not confirm:
        raise RefusedError("dry run: launching needs confirm=True (CLI: --yes)")
    audience = client.audience.get_audience_by_id(cfg.rapidata.audience_id)
    job = audience.assign_job(jd)
    # Record the job before anything can fail: losing the id would mean paying for a new job.
    refs = datapoint_refs(samples)
    _save_job_record(cfg, {
        "job_id": job.id,
        "assigned_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "estimate": est,
        "samples": {s.id: ref for s, ref in zip(samples, refs)},
    })
    for s, ref in zip(samples, refs):
        s.live_datapoint_ref = ref
    save_manifest(cfg, load_manifest_merged(cfg, samples))
    return job.id


def load_manifest_merged(cfg: Config, updated: list[Sample]) -> list[Sample]:
    """The full manifest with `updated` samples swapped in (run_job only sees a slice)."""
    from .data import load_manifest

    by_id = {s.id: s for s in updated}
    return [by_id.get(s.id, s) for s in load_manifest(cfg)]


def job_status(cfg: Config, job_id: str | None = None) -> dict[str, Any]:
    """Non-blocking status snapshot of the recorded (or given) job."""
    job_id = job_id or load_job_record(cfg)["job_id"]
    progress = _client().job.get_job_by_id(job_id).get_progress()
    return {"job_id": job_id, "status": str(progress.state),
            "completion_percentage": float(progress.completion_percentage)}


def collect_results(cfg: Config, job_id: str | None = None, wait: bool = False) -> Path:
    """Save the final results of the recorded (or given) job to results/raw_locate.json.

    With `wait=False` (default) raises JobNotReadyError unless the job has completed; with
    `wait=True` blocks until it completes, but still raises for held states that never finish
    on their own (manual review, out of funds, paused, blocked).
    """
    job_id = job_id or load_job_record(cfg)["job_id"]
    job = _client().job.get_job_by_id(job_id)
    status = str(job.get_status())
    if status in HELD_STATUSES or (not wait and status != "Completed"):
        raise JobNotReadyError(status)
    try:
        results = job.get_results()
    except Exception as e:  # the SDK raises a plain Exception for held states
        raise JobNotReadyError(str(job.get_status()), str(e).splitlines()[0]) from e
    out = cfg.paths.results_dir / cfg.rapidata.raw_results_file
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        json.dump(_to_jsonable(results), fh, indent=2)
    return out


def _to_jsonable(obj: Any) -> Any:
    """SDK results may be dicts, pydantic models or custom objects; coerce to plain JSON."""
    if isinstance(obj, (dict, list, str, int, float, bool)) or obj is None:
        return obj
    for attr in ("to_dict", "model_dump", "dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return fn()
            except TypeError:
                continue
    if hasattr(obj, "__dict__"):
        return {k: _to_jsonable(v) for k, v in vars(obj).items() if not k.startswith("_")}
    return str(obj)


# --------------------------------------------------------------------------- parsing results


def _walk(obj: Any) -> Iterator[Any]:
    yield obj
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v)


def extract_taps(entry: dict[str, Any]) -> tuple[list[Point], list[float]]:
    """Taps and per-tap weights from one Locate result entry, in normalised coordinates.

    Schema of the first live job (Locate results v4.1.0, SDK 3.27): `detailedResults` has one
    item per response, each with `locations: [{"x", "y"}]` in **percent** of the image's width
    and height (0-100) and `userDetails.userScores.global`, the labeller's reliability (0-1),
    used as the weight of each of their taps. `aggregatedResults` repeats the union of those
    locations and is ignored; counting both would double every tap.

    The percent reading was checked against the dataset's independent crowd heatmaps for the
    same 12 images: mean cc 0.60 read as percent, -0.07 read as pixels.
    """
    responses = entry.get("detailedResults")
    if responses is None:
        raise ValueError("result entry has no 'detailedResults'; run --dump-schema")
    pts: list[Point] = []
    weights: list[float] = []
    for r in responses:
        score = (r.get("userDetails") or {}).get("userScores", {}).get("global", 1.0)
        for loc in r.get("locations") or []:
            pts.append((float(loc["x"]) / 100.0, float(loc["y"]) / 100.0))
            weights.append(float(score))
    return pts, weights


def result_entries(raw: dict[str, Any]) -> list[Any]:
    entries = raw.get("results")
    if isinstance(entries, list):
        return entries
    raise ValueError("raw results have no top-level 'results' list; run --dump-schema")


def _entry_strings(entry: Any) -> list[str]:
    """Every string in the entry — dict *keys* included, because `assetUrls` keys filenames."""
    out: list[str] = []
    for node in _walk(entry):
        if isinstance(node, str):
            out.append(node)
        elif isinstance(node, dict):
            out.extend(k for k in node if isinstance(k, str))
    return out


def match_results_to_manifest(
    entries: list[Any], samples: list[Sample]
) -> dict[str, Any]:
    """Map sample.id -> result entry. Tries filename/id matching first, then falls back to order."""
    by_id: dict[str, Any] = {}
    unmatched: list[Any] = []
    id_re = re.compile(r"(s\d{4})")
    for e in entries:
        hit = None
        for s in _entry_strings(e):
            m = id_re.search(Path(s).name)
            if m and any(smp.id == m.group(1) for smp in samples):
                hit = m.group(1)
                break
        if hit and hit not in by_id:
            by_id[hit] = e
        else:
            unmatched.append(e)
    if not by_id and len(unmatched) == len(samples):
        # No recognisable ids anywhere; trust submission order (documented fallback).
        return {s.id: e for s, e in zip(samples, unmatched)}
    return by_id


def parse_results(
    cfg: Config, raw_path: Path, samples: list[Sample], sigma_frac: float | None = None
) -> dict[str, int]:
    """Raw JSON -> per-sample points.json + heatmap .npy under results/human/live/."""
    with raw_path.open() as fh:
        raw = json.load(fh)
    entries = result_entries(raw)
    mapping = match_results_to_manifest(entries, samples)
    out_dir = cfg.paths.human_dir / "live"
    out_dir.mkdir(parents=True, exist_ok=True)
    sigma = sigma_frac if sigma_frac is not None else cfg.heatmap.sigma_frac
    counts: dict[str, int] = {}
    for s in samples:
        e = mapping.get(s.id)
        if e is None:
            continue
        pts, weights = extract_taps(e)
        counts[s.id] = len(pts)
        with (out_dir / f"{s.id}.points.json").open("w") as fh:
            json.dump(pts, fh)
        hm = points_to_heatmap(pts, (s.height, s.width), sigma_frac=sigma, weights=weights)
        np.save(out_dir / f"{s.id}.npy", hm)
    return counts


def schema_sketch(obj: Any, depth: int = 0, max_depth: int = 6) -> str:
    """Compact type-tree of a JSON object, for `--dump-schema`. Lists show their first element."""
    pad = "  " * depth
    if depth > max_depth:
        return pad + "…\n"
    if isinstance(obj, dict):
        s = ""
        for k, v in obj.items():
            s += f"{pad}{k}: {type(v).__name__}\n"
            if isinstance(v, (dict, list)):
                s += schema_sketch(v, depth + 1, max_depth)
        return s
    if isinstance(obj, list):
        s = f"{pad}[list len={len(obj)}]\n"
        if obj:
            s += schema_sketch(obj[0], depth + 1, max_depth)
        return s
    return f"{pad}{type(obj).__name__}\n"
