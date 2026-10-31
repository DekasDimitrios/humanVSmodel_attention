"""Tiny FastAPI viewer: serves results/agreement.json, the overlays, and one static page.

Run with `hvma serve` (or `uvicorn viewer.app:app --reload` while hacking on index.html).
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
DATA = ROOT / "data"
STATIC = Path(__file__).resolve().parent / "static"

app = FastAPI(title="human vs model attention")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/agreement")
def agreement() -> JSONResponse:
    p = RESULTS / "agreement.json"
    if not p.exists():
        raise HTTPException(404, "results/agreement.json missing — run `hvma report` first")
    return JSONResponse(content=__import__("json").loads(p.read_text()))


# overlays and figures live under results/, originals under data/
app.mount("/results", StaticFiles(directory=RESULTS, check_dir=False), name="results")
app.mount("/data", StaticFiles(directory=DATA, check_dir=False), name="data")
