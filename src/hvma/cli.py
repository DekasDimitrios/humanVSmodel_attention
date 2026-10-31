"""`hvma` command-line interface. See CLAUDE.md → Commands."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from .config import Config, load_config, have_rapidata_credentials

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="Human vs. model attention on AI-generated images.")

_state: dict[str, Config] = {}


def cfg() -> Config:
    return _state["cfg"]


@app.callback()
def _main(config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c",
                                      help="YAML config path")) -> None:
    _state["cfg"] = load_config(config)


# --------------------------------------------------------------------------- data


@app.command()
def fetch(n: int = typer.Option(None, "--n", help="samples to stream (default from config)")) -> None:
    """Stream N samples from the Hugging Face dataset into data/."""
    from .data import fetch as _fetch

    samples = _fetch(cfg(), n)
    with_hm = sum(s.has_dataset_heatmap for s in samples)
    typer.echo(f"saved {len(samples)} samples to {cfg().paths.images_dir} "
               f"({with_hm} with dataset heatmaps) → {cfg().paths.manifest}")


@app.command("inspect-dataset")
def inspect_dataset() -> None:
    """Print type/shape of every column of one raw dataset row (resolves CLAUDE.md uncertainty #1)."""
    from .data import inspect_first_row

    for k, v in inspect_first_row(cfg()).items():
        typer.echo(f"{k:28s} {v}")


# --------------------------------------------------------------------------- model


@app.command()
def saliency(limit: int = typer.Option(None, help="only the first N samples")) -> None:
    """Compute CLIP Grad-CAM + attention rollout (and the edge-bias ablation maps) per image."""
    import numpy as np

    from .data import load_image, load_manifest
    from .model import ClipSaliency, ClsAttention, pick_device
    from .report import saliency_path

    c = cfg()
    samples = load_manifest(c)[:limit]
    typer.echo(f"loading {c.model.name} …")
    sal = ClipSaliency(c.model, c.ablation)
    c.paths.saliency_dir.mkdir(parents=True, exist_ok=True)
    scores, outliers, n_maps = {}, {}, 0
    with typer.progressbar(samples, label="clip") as bar:
        for s in bar:
            out = sal(load_image(s))
            for method, grid in out.maps.items():
                np.save(saliency_path(c, s, method), grid)
                n_maps += 1
            scores[s.id] = out.score
            outliers[s.id] = out.n_outliers
    with (c.paths.saliency_dir / "artefactness_scores.json").open("w") as fh:
        json.dump(scores, fh, indent=1)
    if c.ablation.enabled:
        with (c.paths.saliency_dir / "outlier_tokens.json").open("w") as fh:
            json.dump(outliers, fh, indent=1)
        del sal  # free GPU memory before loading the next models
        for name, model_id in c.ablation.register_models.items():
            typer.echo(f"loading {model_id} …")
            attn = ClsAttention(model_id, pick_device(c.model.device))
            with typer.progressbar(samples, label=name) as bar:
                for s in bar:
                    np.save(saliency_path(c, s, f"{name}_attn"), attn(load_image(s)))
                    n_maps += 1
            del attn
    typer.echo(f"wrote {n_maps} maps for {len(samples)} images to {c.paths.saliency_dir}")


# --------------------------------------------------------------------------- rapidata


def _live_samples(limit: int | None):
    from .data import load_manifest

    samples = load_manifest(cfg())
    cap = cfg().rapidata.max_datapoints
    return samples[: min(limit or cap, cap)]


def _require_creds() -> None:
    if not have_rapidata_credentials():
        typer.secho("RAPIDATA_CLIENT_ID / RAPIDATA_CLIENT_SECRET not set — copy .env.example to .env",
                    fg="red")
        raise typer.Exit(2)


@app.command()
def estimate(limit: int = typer.Option(None, help="submit only the first N samples")) -> None:
    """Cost estimate for the live Locate job. Charges nothing, launches nothing."""
    _require_creds()
    from .rapidata_job import RefusedError, estimate as _estimate

    samples = _live_samples(limit)
    try:
        est = _estimate(cfg(), samples)
    except RefusedError as e:
        typer.secho(f"refused: {e}", fg="red"); raise typer.Exit(1)
    typer.echo(json.dumps(est, indent=2))
    if est["estimated_cost"] > cfg().rapidata.max_cost:
        typer.secho(f"⚠ exceeds rapidata.max_cost={cfg().rapidata.max_cost}; `hvma run` will refuse",
                    fg="yellow")


@app.command()
def run(yes: bool = typer.Option(False, "--yes", help="actually launch the paid job"),
        limit: int = typer.Option(None, help="submit only the first N samples")) -> None:
    """Launch the live Locate job (prints the estimate first; requires --yes)."""
    _require_creds()
    from .rapidata_job import RefusedError, estimate as _estimate, run_job

    samples = _live_samples(limit)
    est = _estimate(cfg(), samples)
    typer.echo(f"estimate: {json.dumps(est)}")
    if not yes:
        typer.secho("dry run — pass --yes to launch", fg="yellow"); raise typer.Exit(0)
    try:
        out = run_job(cfg(), samples, confirm=True)
    except RefusedError as e:
        typer.secho(f"refused: {e}", fg="red"); raise typer.Exit(1)
    typer.echo(f"raw results → {out}")


@app.command("parse-results")
def parse_results(raw: Path = typer.Argument(..., exists=True),
                  dump_schema: bool = typer.Option(False, "--dump-schema",
                                                   help="print the JSON type-tree and exit")) -> None:
    """Raw Locate results JSON → per-sample points + heatmaps under results/human/live/."""
    from .data import load_manifest
    from .rapidata_job import parse_results as _parse, schema_sketch

    if dump_schema:
        with raw.open() as fh:
            typer.echo(schema_sketch(json.load(fh)))
        return
    counts = _parse(cfg(), raw, load_manifest(cfg()))
    typer.echo(f"parsed {len(counts)} samples; points per sample: "
               f"min={min(counts.values()) if counts else 0} max={max(counts.values()) if counts else 0}")
    if counts and max(counts.values()) == 0:
        typer.secho("no points found — run with --dump-schema and fix extract_points()", fg="red")


# --------------------------------------------------------------------------- evaluation


@app.command()
def evaluate(human: str = typer.Option("dataset", help="'dataset' or 'live'")) -> None:
    """Metrics for every (sample, method) → metrics_<human>.csv, summary.csv, ablation.csv."""
    from .report import _write_csv, ablation_summary, main_methods, summarize
    from .report import evaluate as _evaluate

    c = cfg()
    rows = _evaluate(c, human)
    summary = summarize(rows, main_methods(c))
    _write_csv(c.paths.results_dir / "summary.csv", summary)
    typer.echo("main methods                         (results/summary.csv)")
    for agg in summary:
        _echo_row(agg)
    if c.ablation.enabled:
        abl = ablation_summary(c, rows, human)
        _write_csv(c.paths.results_dir / "ablation.csv", abl)
        typer.echo("\nedge-bias ablation                   (results/ablation.csv)")
        for agg in abl:
            _echo_row(agg)


def _echo_row(agg: dict) -> None:
    def f(key: str, spec: str) -> str:
        v = agg.get(key)
        return format(v, spec) if isinstance(v, float) and v == v else "  –  "

    typer.echo(f"  {agg['method']:18s} cc={f('cc', '6.3f')} iou@20={f('iou@20', '.3f')} "
               f"pointing={f('pointing_game', '.2f')} js={f('js_divergence', '.3f')} "
               f"border={f('border_mass', '.2f')}")


@app.command()
def report(human: str = typer.Option("dataset", help="'dataset' or 'live'"),
           method: str = typer.Option("gradcam", help="method for the disagreement gallery")) -> None:
    """Overlays, figures, and results/agreement.json for the viewer."""
    from .report import (ablation_chart, ablation_gallery, ablation_summary, disagreement_gallery,
                         metric_histograms, write_agreement_json, write_overlays)
    from .report import evaluate as _evaluate

    c = cfg()
    rows = _evaluate(c, human)
    write_overlays(c, human)
    typer.echo(f"gallery    → {disagreement_gallery(c, rows, human, method)}")
    typer.echo(f"histograms → {metric_histograms(c, rows)}")
    if c.ablation.enabled:
        typer.echo(f"ablation   → {ablation_chart(c, ablation_summary(c, rows, human))}")
        typer.echo(f"             {ablation_gallery(c, rows, human)}")
    typer.echo(f"viewer     → {write_agreement_json(c, rows, human)}")


@app.command()
def serve(port: int = 8000) -> None:
    """Start the viewer (FastAPI) on http://127.0.0.1:<port>."""
    import uvicorn

    # `viewer/` sits at the repo root and is not part of the installed package, and console
    # scripts don't put the CWD on sys.path, so point uvicorn at the repo root explicitly.
    repo_root = Path(__file__).resolve().parents[2]
    uvicorn.run("viewer.app:app", host="127.0.0.1", port=port, reload=False,
                app_dir=str(repo_root))


if __name__ == "__main__":
    app()
