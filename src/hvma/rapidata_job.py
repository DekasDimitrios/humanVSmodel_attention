"""Rapidata Locate job: build, estimate, run, and parse.

Grounded in the public docs (https://docs.rapidata.ai/examples/locate_job/ and
/cost_estimates/). Two things were NOT documented and are handled defensively:

1. The per-response structure of Locate results. `extract_points` walks the JSON and collects
   any dict that has numeric `x` and `y` keys, treating values <= 1 as normalised and larger
   values as pixels. Run `hvma parse-results --dump-schema` after the first live job and
   tighten this once the real shape is known.
2. How result entries map back to input datapoints. `match_results_to_manifest` tries
   `assetUrls` filenames, any string field containing the sample id, and finally index order.

Guardrails (CLAUDE.md hard rules 1–2): `run_job` refuses without `confirm=True`, refuses above
`max_datapoints`, and refuses when the SDK's estimate exceeds `max_cost`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config
from .data import Sample, save_manifest
from .heatmaps import Point, points_to_heatmap


class RefusedError(RuntimeError):
    """Raised when a guardrail blocks a paid action."""


# --------------------------------------------------------------------------- building & running


def _client():
    from rapidata import RapidataClient  # lazy: keeps the free path import-light

    return RapidataClient()  # reads RAPIDATA_CLIENT_ID / RAPIDATA_CLIENT_SECRET from env


def datapoint_refs(samples: list[Sample]) -> list[str]:
    """What we hand the SDK as `datapoints`.

    The docs show public URLs. The SDK is believed to also accept local paths and upload them;
    if `hvma estimate` fails on local paths, host the PNGs (e.g. a public bucket) and set
    `Sample.live_datapoint_ref` to the URL for each sample.
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
    )


def estimate(cfg: Config, samples: list[Sample]) -> dict[str, Any]:
    """Cost estimate without launching anything. Blocks briefly while Rapidata prices it."""
    client = _client()
    jd = build_job_definition(client, cfg, samples)
    est = jd.estimated_cost
    return {
        "estimated_cost": float(est.estimated_cost),
        "datapoint_count": int(est.datapoint_count),
        "required_responses": int(est.required_responses),
    }


def run_job(cfg: Config, samples: list[Sample], confirm: bool = False) -> Path:
    """Launch the Locate job, wait for completion, save raw results JSON. Returns the path."""
    if not confirm:
        raise RefusedError("run_job requires confirm=True (CLI: --yes)")
    client = _client()
    jd = build_job_definition(client, cfg, samples)
    est = jd.estimated_cost
    cost = float(est.estimated_cost)
    if cost > cfg.rapidata.max_cost:
        raise RefusedError(
            f"estimated cost {cost:.2f} exceeds rapidata.max_cost={cfg.rapidata.max_cost}"
        )
    audience = client.audience.get_audience_by_id(cfg.rapidata.audience_id)
    job = audience.assign_job(jd)
    try:
        job.display_progress_bar()
    except Exception:  # noqa: BLE001 — progress bar is cosmetic (needs a TTY)
        pass
    results = job.get_results()
    raw = _to_jsonable(results)
    out = cfg.paths.results_dir / cfg.rapidata.raw_results_file
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        json.dump(raw, fh, indent=2)
    # Remember which ref each sample was submitted under so parsing can join on it.
    refs = datapoint_refs(samples)
    for s, ref in zip(samples, refs):
        s.live_datapoint_ref = ref
    save_manifest(cfg, samples)
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


def _is_point(d: Any) -> bool:
    return (
        isinstance(d, dict)
        and isinstance(d.get("x"), (int, float))
        and isinstance(d.get("y"), (int, float))
    )


def extract_points(entry: Any, size: tuple[int, int] | None = None) -> list[Point]:
    """All (x, y) points anywhere inside one result entry, as normalised coordinates.

    If any coordinate exceeds 1 we assume pixels and divide by `size=(H, W)`; without `size`
    such points are dropped with a warning in the caller.
    """
    pts: list[Point] = []
    for node in _walk(entry):
        if _is_point(node):
            pts.append((float(node["x"]), float(node["y"])))
    if not pts:
        return pts
    if max(max(x, y) for x, y in pts) > 1.0:
        if size is None:
            return []
        h, w = size
        pts = [(x / w, y / h) for x, y in pts]
    return pts


def extract_weights(entry: Any, n: int) -> list[float] | None:
    """Per-point reliability weights from `userDetails.userScores.global`, if present.

    Only returned when we find exactly one score per point; otherwise None (uniform weights).
    """
    scores: list[float] = []
    for node in _walk(entry):
        if isinstance(node, dict) and "userScores" in node:
            g = node["userScores"].get("global") if isinstance(node["userScores"], dict) else None
            if isinstance(g, (int, float)):
                scores.append(float(g))
    return scores if len(scores) == n else None


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
        pts = extract_points(e, size=(s.height, s.width))
        weights = extract_weights(e, len(pts))
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
