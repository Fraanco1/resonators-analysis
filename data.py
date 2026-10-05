"""Read-only access to QCoDeS SQLite databases, without hardcoding parameter names.

Each run's parameters are classified from its run_description (QCoDeS
interdependencies: depends_on, label, unit):
  * frequency  : the setpoint whose name/label/unit looks like a frequency
  * magnitude  : dependent named/labelled mag/amp/abs/S21 (dB or linear)
  * phase      : dependent named/labelled phase/angle/arg (deg or rad)
  * real/imag  : dependents named real/imag, I/Q, or a complex-valued one
  * outer      : any other setpoint (e.g. power in a 2D sweep) -> one trace per value
  * scalars    : parameters that depend on nothing and nobody depends on
                 (power, temperature, bandwidth, attenuation, ...)
Both array storage (one blob per sweep) and point-per-row storage work.
"""
import io
import json
import re
import sqlite3
from collections import OrderedDict
from pathlib import Path
from threading import Lock

import numpy as np

RE_FREQ = re.compile(r"freq|frequency", re.I)
RE_MAG = re.compile(r"mag|amp|abs|s21|s11|s12|s22|transmission|power_db|lin", re.I)
RE_PHASE = re.compile(r"phase|angle|arg", re.I)
RE_REAL = re.compile(r"(^|[\W_])(re|real|i)($|[\W_])|real", re.I)
RE_IMAG = re.compile(r"(^|[\W_])(im|imag|q)($|[\W_])|imag", re.I)


def connect(db_path):
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
    return con


def find_databases(root):
    root = Path(root)
    if root.is_file():
        return [str(root)]
    return sorted(str(p) for p in root.rglob("*.db"))


def _decode(v):
    """QCoDeS stores arrays as np.save() blobs; numbers come back as-is."""
    if isinstance(v, (bytes, bytearray, memoryview)):
        return np.load(io.BytesIO(bytes(v)), allow_pickle=False)
    return v


def list_experiments(con):
    rows = con.execute(
        "SELECT e.exp_id, e.name, e.sample_name, e.start_time, COUNT(r.run_id) "
        "FROM experiments e LEFT JOIN runs r ON r.exp_id = e.exp_id "
        "GROUP BY e.exp_id ORDER BY e.exp_id"
    ).fetchall()
    return [
        {"exp_id": r[0], "name": r[1], "sample": r[2], "start_time": r[3], "n_runs": r[4]}
        for r in rows
    ]


# ── Parameter classification ──────────────────────────────────────────────────

def _paramspecs(con, run_id):
    row = con.execute(
        "SELECT result_table_name, parameters, run_description FROM runs WHERE run_id=?", (run_id,)
    ).fetchone()
    if row is None:
        raise KeyError(f"Run {run_id} does not exist in this database")
    tbl, params, desc = row
    specs = {}
    try:
        d = json.loads(desc)
        for ps in d["interdependencies"]["paramspecs"]:
            specs[ps["name"]] = ps
    except Exception:
        pass
    if not specs:  # very old databases: fall back to the layouts table
        for name, label, unit in con.execute(
            "SELECT parameter, label, unit FROM layouts WHERE run_id=?", (run_id,)
        ):
            specs[name] = {"name": name, "label": label, "unit": unit, "depends_on": [], "paramtype": ""}
        deps = con.execute(
            "SELECT l1.parameter, l2.parameter FROM dependencies d "
            "JOIN layouts l1 ON l1.layout_id = d.dependent "
            "JOIN layouts l2 ON l2.layout_id = d.independent "
            "WHERE l1.run_id = ? ORDER BY d.axis_num", (run_id,)
        ).fetchall()
        for dep, indep in deps:
            specs[dep]["depends_on"].append(indep)
    for n in (params or "").split(","):
        if n and n not in specs:
            specs[n] = {"name": n, "label": n, "unit": "", "depends_on": [], "paramtype": ""}
    return tbl, specs


def _q(name):
    """Quote an SQL identifier (names come from the database itself, so escape embedded quotes)."""
    return '"' + str(name).replace('"', '""') + '"'


def _text(ps):
    return f'{ps["name"]} {ps.get("label") or ""}'


def classify(specs):
    dependents = {n: ps for n, ps in specs.items() if ps.get("depends_on")}
    # Ordered by first appearance (column order), so the choice below is deterministic.
    setpoints = list(dict.fromkeys(s for ps in dependents.values() for s in ps["depends_on"]))
    scalars = [n for n in specs if n not in dependents and n not in setpoints]

    def is_freq(n):
        ps = specs[n]
        return bool(RE_FREQ.search(_text(ps))) or (ps.get("unit") or "").lower() in ("hz", "khz", "mhz", "ghz")

    freq = next((s for s in setpoints if is_freq(s)), None)
    if freq is None and setpoints:
        freq = setpoints[0]
    outer = sorted(s for s in setpoints if s != freq)

    roles = {"freq": freq, "outer": outer, "scalars": scalars,
             "mag": None, "phase": None, "real": None, "imag": None, "complex": None}
    for n, ps in dependents.items():
        if freq not in ps["depends_on"]:
            continue
        t = _text(ps)
        if ps.get("paramtype") == "complex" and roles["complex"] is None:
            roles["complex"] = n
        elif RE_PHASE.search(t) and roles["phase"] is None:
            roles["phase"] = n
        elif RE_IMAG.search(t) and roles["imag"] is None:
            roles["imag"] = n
        elif RE_REAL.search(t) and roles["real"] is None:
            roles["real"] = n
        elif RE_MAG.search(t) and roles["mag"] is None:
            roles["mag"] = n
    if roles["mag"] is None and roles["complex"] is None and roles["real"] is None:
        # Fall back to the first remaining dependent on frequency.
        used = {roles[k] for k in ("phase", "imag")}
        rest = [n for n, ps in dependents.items() if freq in ps["depends_on"] and n not in used]
        if rest:
            roles["mag"] = rest[0]
    return roles


# ── Run metadata ──────────────────────────────────────────────────────────────

def _scalar_values(con, tbl, specs, names):
    out = {}
    for n in names:
        try:
            row = con.execute(f"SELECT {_q(n)} FROM {_q(tbl)} WHERE {_q(n)} IS NOT NULL LIMIT 1").fetchone()
        except sqlite3.OperationalError:
            continue
        if row is None:
            continue
        v = _decode(row[0])
        if isinstance(v, np.ndarray):
            v = v.ravel()[0] if v.size else None
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        ps = specs[n]
        out[n] = {"value": v, "label": ps.get("label") or n, "unit": ps.get("unit") or ""}
    return out


def list_runs(con, exp_id):
    rows = con.execute(
        "SELECT run_id, name, result_counter, run_timestamp, completed_timestamp, is_completed "
        "FROM runs WHERE exp_id=? ORDER BY run_id", (exp_id,)
    ).fetchall()
    out = []
    for run_id, name, counter, ts, cts, done in rows:
        item = {"run_id": run_id, "name": name, "counter": counter, "timestamp": ts,
                "completed": bool(done), "scalars": {}}
        try:
            tbl, specs = _paramspecs(con, run_id)
            roles = classify(specs)
            item["scalars"] = _scalar_values(con, tbl, specs, roles["scalars"])
        except Exception as e:  # keep listing even if one run is odd
            item["error"] = str(e)
        out.append(item)
    return out


# ── Loading traces ────────────────────────────────────────────────────────────

def _long_form(con, tbl, name, setpoints):
    """All values of `name` with its setpoints, flattened (works for array and numeric storage)."""
    cols = [name] + list(setpoints)
    sel = ", ".join(_q(c) for c in cols)
    rows = con.execute(f"SELECT {sel} FROM {_q(tbl)} WHERE {_q(name)} IS NOT NULL ORDER BY id").fetchall()
    chunks = [[] for _ in cols]
    for row in rows:
        vals = [_decode(v) for v in row]
        if any(v is None for v in vals):
            continue
        arrs = np.broadcast_arrays(*[np.asarray(v) for v in vals])
        for i, a in enumerate(arrs):
            chunks[i].append(np.ravel(a))
    if not chunks[0]:
        return [np.array([]) for _ in cols]
    return [np.concatenate(c) for c in chunks]


def _unit_is_db(ps, values):
    u = (ps.get("unit") or "").lower()
    if "db" in u:
        return True
    if u in ("", "a.u.", "au", "v", "w"):
        # No unit hint: negative values are not possible for a linear magnitude.
        return bool(np.nanmin(values) < 0) if u == "" else False
    return False


def _phase_to_deg(ps, values):
    u = (ps.get("unit") or "").lower()
    if u.startswith("deg") or u == "°":
        return values
    if u.startswith("rad"):
        return np.rad2deg(values)
    return values if np.nanmax(np.abs(values)) > 2 * np.pi + 0.5 else np.rad2deg(values)


class RunData:
    def __init__(self, run_id, roles, specs, traces, scalars):
        self.run_id = run_id
        self.roles = roles
        self.specs = specs
        self.traces = traces      # list of dicts: freq, mag_db, phase_deg, outer
        self.scalars = scalars


def _select(sp, key):
    """Mask of long-form rows whose outer setpoints equal `key`."""
    mask = np.ones(len(sp[0]), bool)
    for j, o in enumerate(sp[1:]):
        mask &= np.isclose(o.astype(float), key[j], rtol=1e-9, atol=1e-12)
    return mask


def load_run(con, run_id):
    tbl, specs = _paramspecs(con, run_id)
    roles = classify(specs)
    f = roles["freq"]
    if f is None:
        raise ValueError("Could not identify a frequency setpoint in this run.")
    order = [f] + roles["outer"]

    def get(name):
        """(values, [freq, outer...]) for a dependent parameter."""
        sps = [s for s in order if s in specs[name]["depends_on"]]
        v, *sp = _long_form(con, tbl, name, sps)
        return v, sp

    # Magnitude (dB) and phase (deg) in long form, possibly on different grids.
    ph_lf = None
    if roles["complex"] or (roles["real"] and roles["imag"]):
        if roles["complex"]:
            v, sp = get(roles["complex"])
            z = v.astype(complex)
        else:
            re_, sp = get(roles["real"])
            im_, _ = get(roles["imag"])
            z = re_.astype(float) + 1j * im_.astype(float)
        mag = 20 * np.log10(np.abs(z))
        ph_lf = (np.rad2deg(np.angle(z)), sp)
    else:
        if roles["mag"] is None:
            raise ValueError("Could not identify a magnitude parameter in this run.")
        mag, sp = get(roles["mag"])
        mag = mag.astype(float)
        if not _unit_is_db(specs[roles["mag"]], mag):
            mag = 20 * np.log10(np.abs(mag))
        if roles["phase"]:
            ph, sp_ph = get(roles["phase"])
            ph_lf = (_phase_to_deg(specs[roles["phase"]], ph.astype(float)), sp_ph)
    sp = [a.astype(float) for a in sp]
    freq = sp[0]
    n_outer = len(sp) - 1

    # One trace per combination of outer setpoints (e.g. each power in a 2D sweep).
    if n_outer:
        uniq = np.unique(np.stack(sp[1:], axis=1), axis=0)
    else:
        uniq = np.zeros((1, 0))

    traces = []
    for key in uniq:
        m_mask = _select(sp, key)
        idx = np.where(m_mask)[0]
        idx = idx[np.argsort(freq[idx], kind="stable")]
        fr, m = freq[idx], mag[idx]
        p = None
        if ph_lf is not None:
            pv, psp = ph_lf
            psp = [a.astype(float) for a in psp]
            pidx = np.where(_select(psp, key) if len(psp) > 1 else np.ones(len(pv), bool))[0]
            pidx = pidx[np.argsort(psp[0][pidx], kind="stable")]
            pf, pp = psp[0][pidx], pv[pidx]
            if len(pf) == len(fr) and np.allclose(pf, fr):
                p = pp
            elif len(pf) > 1:
                p = np.interp(fr, pf, pp)
            if p is not None:
                p = np.rad2deg(np.unwrap(np.deg2rad(p)))
        good = np.isfinite(fr) & np.isfinite(m)
        if p is not None:
            good &= np.isfinite(p)
        outer = {}
        for j, name in enumerate(roles["outer"][:n_outer]):
            outer[name] = {"value": float(key[j]), "label": specs[name].get("label") or name,
                           "unit": specs[name].get("unit") or ""}
        traces.append({
            "freq": fr[good],
            "mag_db": m[good],
            "phase_deg": p[good] if p is not None else None,
            "outer": outer,
        })

    scalars = _scalar_values(con, tbl, specs, roles["scalars"])
    return RunData(run_id, roles, specs, traces, scalars)


class RunCache:
    """Small LRU cache of decoded runs, keyed by (db, run_id)."""

    def __init__(self, size=12):
        self.size = size
        self._d = OrderedDict()
        self._lock = Lock()

    def get(self, db, con, run_id):
        key = (db, run_id)
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key)
                return self._d[key]
        rd = load_run(con, run_id)
        with self._lock:
            self._d[key] = rd
            while len(self._d) > self.size:
                self._d.popitem(last=False)
        return rd
