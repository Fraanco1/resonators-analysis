"""Fits of synthetic resonances with known parameters.

    pytest tests/            (about a minute)

Checks, for notch and reflection ports over a range of coupling regimes:
  * noise-free data is recovered exactly (fr, QL, Qi, Qc, cable delay);
  * with noise, the fits are unbiased and the reported errors match the real
    scatter (pull = (fit - truth) / error has unit standard deviation);
  * auto-detection finds every resonance of a multi-resonance trace;
  * the whole path through the web API (QCoDeS database -> /api/autofit) works.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from fitting import detect_resonances, fit_resonance  # noqa: E402
from synthetic import Environment, Resonance, trace, window  # noqa: E402

warnings.filterwarnings("ignore")

CASES = {
    "critical": Resonance(Qi=1e5, Qc=1e5),
    "under-coupled x10": Resonance(Qi=1e6, Qc=1e5),
    "over-coupled x10": Resonance(Qi=1e4, Qc=1e5),
    "over-coupled x30": Resonance(Qi=3e4, Qc=1e3),
    "asymmetric phi=+0.4": Resonance(Qi=2e5, Qc=1e5, phi=0.4),
    "asymmetric phi=-0.6": Resonance(Qi=2e5, Qc=1e5, phi=-0.6),
    "low Q": Resonance(fr=4e9, Qi=3e3, Qc=2e3),
    "high Q": Resonance(fr=7e9, Qi=2e6, Qc=1e6),
    "reflection critical": Resonance(Qi=5e4, Qc=5e4, port="reflection"),
    "reflection over-coupled": Resonance(Qi=2e5, Qc=2e4, port="reflection"),
}


def _fit(res, env, rng, **kw):
    f = window(res)
    mag, ph = trace(f, [res], env, rng)
    return fit_resonance(f, mag, ph, f[0], f[-1], port=res.port, **kw)


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("delay", [40e-9, 0.0])
def test_noise_free_exact(name, delay):
    res = CASES[name]
    o = _fit(res, Environment(snr_db=None, delay=delay), np.random.default_rng(0))
    assert o["method"] == "full model"
    assert abs(o["fr"] - res.fr) < 1e-4 * res.fwhm
    for k in ("QL", "Qi", "Qc"):
        assert o[k] == pytest.approx(getattr(res, k), rel=1e-4), k
    assert o["delay"] == pytest.approx(delay, abs=1e-12)
    assert not o["flags"]


@pytest.mark.parametrize("name", ["critical", "asymmetric phi=+0.4", "high Q", "reflection critical"])
def test_noise_free_circle_fit_only(name):
    """The plain resonator_tools circle fit (refine=False) must also be right on clean data."""
    res = CASES[name]
    o = _fit(res, Environment(snr_db=None), np.random.default_rng(0), refine=False)
    assert o["method"] == "circle"
    for k in ("QL", "Qi", "Qc"):
        assert o[k] == pytest.approx(getattr(res, k), rel=1e-3), k


def test_fixed_delay_is_used():
    res = CASES["critical"]
    o = _fit(res, Environment(snr_db=None), np.random.default_rng(0), electric_delay=40e-9)
    assert o["delay"] == pytest.approx(40e-9, abs=1e-15)
    assert o["Qi"] == pytest.approx(res.Qi, rel=1e-4)


@pytest.mark.parametrize("name", ["critical", "under-coupled x10", "asymmetric phi=+0.4",
                                  "high Q", "over-coupled x30", "reflection over-coupled"])
def test_noisy_unbiased_with_honest_errors(name):
    res = CASES[name]
    rng = np.random.default_rng(1)
    n = 60
    vals = {k: [] for k in ("QL", "Qi", "Qc", "fr")}
    pulls = {k: [] for k in vals}
    ratio = []
    for _ in range(n):
        o = _fit(res, Environment(snr_db=30), rng)
        ratio.append(o["resid_noise"])
        for k in vals:
            vals[k].append(o[k])
            pulls[k].append((o[k] - getattr(res, k)) / o[k + "_err"])
    for k in ("QL", "Qi", "Qc"):
        v = np.array(vals[k])
        scatter = np.std(v)
        # Mean within ~3 standard errors of the truth (no significant bias).
        assert abs(np.mean(v) - getattr(res, k)) < 3.5 * scatter / np.sqrt(n) + 0.002 * getattr(res, k), k
        # Reported errors match the scatter: pull std ~ 1.
        assert 0.7 < np.std(pulls[k]) < 1.4, (k, np.std(pulls[k]))
    assert abs(np.mean(vals["fr"]) - res.fr) < 0.05 * res.fwhm
    # Residuals sit at the noise level when the model is right.
    assert 0.85 < np.median(ratio) < 1.15


def test_noisy_over_coupled_robust():
    """Strongly over-coupled notch: the circle fit alone sometimes misses; the result must not."""
    res = CASES["over-coupled x10"]
    rng = np.random.default_rng(2)
    bad = 0
    for _ in range(60):
        o = _fit(res, Environment(snr_db=30), rng)
        bad += abs(o["QL"] / res.QL - 1) > 0.5 or abs(o["fr"] - res.fr) > res.fwhm
    assert bad <= 2


def kerr_notch(f, res, xi):
    """Notch with a Kerr (Duffing) nonlinearity: the detuning is pulled by the stored energy.

    x_eff = x - xi * n(x_eff), n = 1 / (1 + 4 x_eff^2), x in linewidths. Below the
    bifurcation (xi < ~0.77) this gives the familiar shark-fin dip of driven resonators.
    """
    x = (f - res.fr) / res.fwhm
    xe = x.copy()
    for _ in range(500):                           # damped fixed-point iteration
        xe = 0.7 * xe + 0.3 * (x - xi / (1 + 4 * xe ** 2))
    return 1 - (res.QL / res.Qc) / (1 + 2j * xe)


def test_kerr_distorted_line_is_flagged():
    """A non-Lorentzian (Kerr-pulled) dip must be flagged, not reported silently."""
    res = Resonance(Qi=2e5, Qc=1e5)
    f = window(res)
    z = Environment(snr_db=40).apply(f, kerr_notch(f, res, xi=0.6), np.random.default_rng(0))
    o = fit_resonance(f, 20 * np.log10(abs(z)), np.rad2deg(np.unwrap(np.angle(z))), f[0], f[-1])
    assert o["resid_noise"] > 3
    assert any("residuals" in fl for fl in o["flags"])
    # ...while the same resonance without the nonlinearity is clean.
    z0 = Environment(snr_db=40).apply(f, kerr_notch(f, res, xi=0.0), np.random.default_rng(0))
    o0 = fit_resonance(f, 20 * np.log10(abs(z0)), np.rad2deg(np.unwrap(np.angle(z0))), f[0], f[-1])
    assert not o0["flags"] and o0["Qi"] == pytest.approx(res.Qi, rel=0.05)


def test_detect_and_fit_multi_resonance_trace():
    rng = np.random.default_rng(3)
    truth = [Resonance(fr=7.02e9, Qi=1e5, Qc=2e5), Resonance(fr=7.10e9, Qi=3e4, Qc=5e4),
             Resonance(fr=7.105e9, Qi=4e5, Qc=1e5, phi=0.3), Resonance(fr=7.30e9, Qi=1e4, Qc=3e4),
             Resonance(fr=7.45e9, Qi=5e4, Qc=1e5)]
    f = np.linspace(7.0e9, 7.5e9, 100001)
    env = Environment(snr_db=35)
    mag, ph = trace(f, truth, env, rng)
    # Standing-wave ripple on the baseline, as in real cabling.
    mag = mag + 0.4 * np.sin(2 * np.pi * f / 37e6)
    det = detect_resonances(f, mag)["resonances"]
    assert len(det) == len(truth)
    for d, r in zip(det, truth):
        assert abs(d["f0"] - r.fr) < r.fwhm
        o = fit_resonance(f, mag, ph, d["f1"], d["f2"])
        assert o["fr"] == pytest.approx(r.fr, abs=0.05 * r.fwhm)
        assert o["QL"] == pytest.approx(r.QL, rel=0.1)
        assert o["Qc"] == pytest.approx(r.Qc, rel=0.1)


def test_end_to_end_through_api(tmp_path, monkeypatch):
    """Write a QCoDeS database with known resonances and fit it through the web API."""
    qcodes = pytest.importorskip("qcodes")
    from qcodes.dataset import Measurement, initialise_or_create_database_at, load_or_create_experiment
    from qcodes.parameters import Parameter

    db = tmp_path / "synthetic.db"
    initialise_or_create_database_at(str(db))
    exp = load_or_create_experiment("synthetic", sample_name="test")
    fa = Parameter("vna_frequency_axis", unit="Hz", label="Frequency", set_cmd=None, get_cmd=None)
    mag = Parameter("vna_tr1_magnitude", unit="dB", label="Magnitude", set_cmd=None, get_cmd=None)
    ph = Parameter("vna_tr2_unwrapped_phase", unit="deg", label="Phase", set_cmd=None, get_cmd=None)
    pw = Parameter("vna_power", unit="dBm", set_cmd=None, get_cmd=None)
    meas = Measurement(exp=exp)
    meas.register_parameter(fa, paramtype="array")
    meas.register_parameter(mag, setpoints=(fa,), paramtype="array")
    meas.register_parameter(ph, setpoints=(fa,), paramtype="array")
    meas.register_parameter(pw)
    truth = [Resonance(fr=5.03e9, Qi=5e5, Qc=1e5), Resonance(fr=5.10e9, Qi=2e5, Qc=7e4, phi=-0.2)]
    f = np.linspace(5.0e9, 5.15e9, 60001)
    rng = np.random.default_rng(4)
    for power in (-30.0, -20.0):
        m, p = trace(f, truth, Environment(snr_db=40), rng)
        with meas.run() as ds:
            ds.add_result((fa, f), (mag, m), (ph, p))
            ds.add_result((pw, power))

    import app as webapp
    from fastapi.testclient import TestClient
    monkeypatch.setattr(webapp, "DB_ROOT", str(tmp_path))
    client = TestClient(webapp.app)

    dbs = client.get("/api/databases").json()["databases"]
    assert [d["path"] for d in dbs] == [str(db)]
    runs = client.get("/api/runs", params={"db": str(db), "exp_id": 1}).json()
    assert [r["scalars"]["vna_power"]["value"] for r in runs] == [-30.0, -20.0]

    r = client.post("/api/autofit", json={"db": str(db), "run_id": runs[0]["run_id"]}).json()
    fits = [x for x in r["results"] if x["ok"]]
    assert len(fits) == len(truth)
    for o, t in zip(fits, truth):
        assert o["fr"] == pytest.approx(t.fr, abs=0.05 * t.fwhm)
        assert o["Qi"] == pytest.approx(t.Qi, rel=0.15)
        assert o["Qc"] == pytest.approx(t.Qc, rel=0.05)
        assert not o["flags"]

    # Batch path: reuse those windows on the second run.
    windows = [[o["f1"], o["f2"]] for o in fits]
    r2 = client.post("/api/autofit", json={"db": str(db), "run_id": runs[1]["run_id"], "windows": windows,
                                           "include_model": False}).json()
    assert all(x["ok"] for x in r2["results"])
    assert "model" not in r2["results"][0]
