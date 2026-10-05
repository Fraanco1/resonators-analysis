"""Resonator fit web app.

    python app.py                      # scans ../Database for *.db files
    python app.py --db /path/to/dir    # or a directory / a single .db file
    python app.py --port 8050 --host 0.0.0.0

The databases are opened read-only.
"""
import argparse
import base64
import os
import sqlite3
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import data
from fitting import detect_resonances, fit_resonance, recenter_window

HERE = Path(__file__).resolve().parent
DB_ROOT = os.environ.get("RESONATOR_DB", str(HERE.parent / "Database"))

app = FastAPI(title="Resonator fits")
cache = data.RunCache()


_db_list = {"root": None, "t": 0.0, "dbs": []}


def _databases(max_age=10.0):
    """*.db files under DB_ROOT; the directory walk is cached for a few seconds."""
    now = time.monotonic()
    if _db_list["root"] != DB_ROOT or now - _db_list["t"] > max_age:
        _db_list.update(root=DB_ROOT, t=now, dbs=data.find_databases(DB_ROOT))
    return _db_list["dbs"]


def _db_path(db: str) -> str:
    """Only databases found under DB_ROOT may be opened."""
    dbs = _databases()
    if db in dbs:
        return db
    raise HTTPException(404, f"Unknown database: {db}")


def _run(db: str, run_id: int):
    path = _db_path(db)
    con = data.connect(path)
    try:
        return cache.get(path, con, run_id)
    except KeyError as e:
        raise HTTPException(404, str(e.args[0]) if e.args else "Not found")
    except (ValueError, sqlite3.Error) as e:
        raise HTTPException(422, f"Could not read run {run_id}: {e}")
    finally:
        con.close()


def _trace(db, run_id, trace):
    rd = _run(db, run_id)
    if not 0 <= trace < len(rd.traces):
        raise HTTPException(404, f"Run {run_id} has {len(rd.traces)} trace(s)")
    return rd, rd.traces[trace]


def _b64(a, dtype):
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode()


# ── Browsing ──────────────────────────────────────────────────────────────────

@app.get("/api/databases")
def databases():
    root = Path(DB_ROOT)
    return {"root": str(root),
            "databases": [{"path": p, "name": str(Path(p).relative_to(root)) if root.is_dir() else Path(p).name}
                          for p in _databases()]}


@app.get("/api/experiments")
def experiments(db: str):
    con = data.connect(_db_path(db))
    try:
        return data.list_experiments(con)
    finally:
        con.close()


@app.get("/api/runs")
def runs(db: str, exp_id: int):
    con = data.connect(_db_path(db))
    try:
        return data.list_runs(con, exp_id)
    finally:
        con.close()


@app.get("/api/run")
def run(db: str, run_id: int, trace: int = 0):
    rd, tr = _trace(db, run_id, trace)
    roles = {k: v for k, v in rd.roles.items() if k != "scalars"}
    return {
        "run_id": run_id,
        "roles": roles,
        "n_traces": len(rd.traces),
        "traces": [t["outer"] for t in rd.traces],
        "trace": trace,
        "scalars": rd.scalars,
        "outer": tr["outer"],
        "n_points": int(len(tr["freq"])),
        "has_phase": tr["phase_deg"] is not None,
        # Binary-packed arrays keep a 100k-point sweep at ~1.3 MB instead of ~5 MB of JSON.
        "freq": _b64(tr["freq"], "<f8"),
        "mag": _b64(tr["mag_db"], "<f4"),
        "phase": _b64(tr["phase_deg"], "<f4") if tr["phase_deg"] is not None else None,
    }


# ── Fitting ───────────────────────────────────────────────────────────────────

class FitOptions(BaseModel):
    port: str = "notch"
    guessdelay: bool = True
    electric_delay_ns: Optional[float] = None


class DetectOptions(BaseModel):
    min_depth_db: Optional[float] = None
    window_factor: float = 8.0
    baseline_points: Optional[int] = None


class FitRequest(BaseModel):
    db: str
    run_id: int
    trace: int = 0
    f1: float
    f2: float
    fit: FitOptions = FitOptions()


class DetectRequest(BaseModel):
    db: str
    run_id: int
    trace: int = 0
    detect: DetectOptions = DetectOptions()


class AutoFitRequest(BaseModel):
    db: str
    run_id: int
    trace: int = 0
    detect: DetectOptions = DetectOptions()
    fit: FitOptions = FitOptions()
    # If given, fit these windows instead of detecting (e.g. reuse one run's windows on
    # every run of a power sweep); recenter shifts each onto the local minimum first.
    windows: Optional[List[List[float]]] = Field(None, max_length=500)
    recenter: bool = True
    include_model: bool = True


def _fit_kwargs(o: FitOptions):
    return dict(port=o.port, guessdelay=o.guessdelay,
                electric_delay=None if o.electric_delay_ns is None else o.electric_delay_ns * 1e-9)


@app.post("/api/fit")
def fit(req: FitRequest):
    _, tr = _trace(req.db, req.run_id, req.trace)
    try:
        return fit_resonance(tr["freq"], tr["mag_db"], tr["phase_deg"], req.f1, req.f2, **_fit_kwargs(req.fit))
    except Exception as e:
        raise HTTPException(422, f"Fit failed: {e}")


@app.post("/api/detect")
def detect(req: DetectRequest):
    _, tr = _trace(req.db, req.run_id, req.trace)
    return detect_resonances(tr["freq"], tr["mag_db"], **req.detect.model_dump())


@app.post("/api/autofit")
def autofit(req: AutoFitRequest):
    rd, tr = _trace(req.db, req.run_id, req.trace)
    f, m, p = tr["freq"], tr["mag_db"], tr["phase_deg"]
    if req.windows is None:
        det = detect_resonances(f, m, **req.detect.model_dump())
        windows = [(r["f1"], r["f2"]) for r in det["resonances"]]
    else:
        det = None
        windows = [tuple(w) for w in req.windows]
        if req.recenter:
            windows = [recenter_window(f, m, a, b) for a, b in windows]
    results = []
    for i, (a, b) in enumerate(windows):
        try:
            r = fit_resonance(f, m, p, a, b, **_fit_kwargs(req.fit))
            if not req.include_model:
                r.pop("model")
            r["ok"] = True
        except Exception as e:
            r = {"ok": False, "error": str(e), "f1": a, "f2": b}
        r["index"] = i
        results.append(r)
    return {"detection": det, "results": results, "n_traces": len(rd.traces), "outer": tr["outer"]}


# ── Static frontend ───────────────────────────────────────────────────────────

app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=DB_ROOT, help="Database directory (scanned for *.db) or a single .db file")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8050)
    args = ap.parse_args()
    DB_ROOT = str(Path(args.db).expanduser().resolve())
    print(f"Databases under {DB_ROOT}: {len(_databases())} found -> http://{args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port)
