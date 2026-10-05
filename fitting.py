"""Resonance detection and notch-port circle fits (resonator_tools, as in RFP-Software)."""
import warnings

import matplotlib

matplotlib.use("Agg")  # resonator_tools imports pyplot; keep it headless

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks, peak_widths

from resonator_tools import circuit


def to_complex(mag_db, phase_deg):
    return 10 ** (np.asarray(mag_db) / 20) * np.exp(1j * np.deg2rad(phase_deg))


def _finite(x):
    """JSON-safe float (NaN/inf -> None)."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def fit_resonance(freq, mag_db, phase_deg, f1, f2, port="notch", guessdelay=True,
                  electric_delay=None, max_model_points=2000):
    """Fit the window [f1, f2] with resonator_tools (circuit.notch_port or reflection_port).

    Returns fr, QL, Qi, |Qc| with errors, plus the model curve (magnitude dB /
    phase deg) over the window. For notch, Qi is the diameter-corrected value
    (Qi_dia_corr), as in RFP-Software.
    """
    if phase_deg is None:
        raise ValueError("This trace has no phase data; a circle fit needs magnitude and phase.")
    freq = np.asarray(freq)
    sel = (freq >= min(f1, f2)) & (freq <= max(f1, f2))
    if sel.sum() < 20:
        raise ValueError(f"Only {int(sel.sum())} points in the selected range; select a wider range.")
    f = freq[sel]
    z = to_complex(np.asarray(mag_db)[sel], np.asarray(phase_deg)[sel])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if port == "reflection":
            p = circuit.reflection_port(f, z)
            p.autofit(electric_delay=electric_delay)
        else:
            p = circuit.notch_port(f, z)
            p.autofit(electric_delay=electric_delay, guessdelay=guessdelay)
    r = p.fitresults
    if not r or not np.isfinite(r.get("fr", np.nan)):
        raise RuntimeError("Fit did not converge.")
    if port == "reflection":
        qi, qi_err, qc, qc_err = r.get("Qi"), r.get("Qi_err"), r.get("Qc"), r.get("Qc_err")
        delay = getattr(p, "_delay", None)
    else:
        qi, qi_err = r.get("Qi_dia_corr"), r.get("Qi_dia_corr_err")
        qc, qc_err = r.get("absQc"), r.get("absQc_err")
        delay = r.get("delay")

    # Keep the model curve light enough to plot.
    step = max(1, len(f) // max_model_points)
    zs = p.z_data_sim[::step]
    model_phase = np.rad2deg(np.unwrap(np.angle(zs)))
    # Align the model's phase branch with the data's unwrapped phase.
    data_phase = np.asarray(phase_deg)[sel][::step]
    model_phase += 360 * np.round(np.median(data_phase - model_phase) / 360)

    flags = []
    if qi is not None and qi <= 0:
        flags.append("Qi ≤ 0 (unphysical; nonlinear or bad window?)")
    for name, v, e in (("Qi", qi, qi_err), ("Qc", qc, qc_err), ("QL", r.get("Ql"), r.get("Ql_err"))):
        if v is not None and e is not None and np.isfinite(e) and abs(e) > abs(v):
            flags.append(f"{name} error > 100%")
    if not (f[0] <= r["fr"] <= f[-1]):
        flags.append("fr outside window")

    return {
        "flags": flags,
        "port": port,
        "f1": float(f[0]),
        "f2": float(f[-1]),
        "n_points": int(len(f)),
        "fr": _finite(r.get("fr")),
        "fr_err": _finite(r.get("fr_err")),
        "QL": _finite(r.get("Ql")),
        "QL_err": _finite(r.get("Ql_err")),
        "Qi": _finite(qi),
        "Qi_err": _finite(qi_err),
        "Qc": _finite(qc),
        "Qc_err": _finite(qc_err),
        "phi0": _finite(r.get("phi0")),
        "chi_square": _finite(r.get("chi_square")),
        "delay": _finite(delay),
        "model": {
            "freq": f[::step].tolist(),
            "mag": (20 * np.log10(np.abs(zs))).tolist(),
            "phase": model_phase.tolist(),
        },
    }


def _baseline(mag, win):
    """Running median; for windows comparable to the trace use a line through the edges."""
    n = len(mag)
    if win >= n // 3:
        k = max(5, n // 10)
        x = np.r_[np.arange(k), np.arange(n - k, n)]
        coef = np.polyfit(x, mag[x], 1)
        return np.polyval(coef, np.arange(n))
    return median_filter(mag, size=win, mode="nearest")


def detect_resonances(freq, mag_db, min_depth_db=None, window_factor=8.0, baseline_points=None,
                      max_count=500):
    """Find dips and a fit window for each one.

    The baseline is removed with a running median (window grown automatically
    until it is much wider than the widest dip, unless baseline_points is
    given), dips are found with scipy.signal.find_peaks on the depth below the
    baseline, and each fit window is +-window_factor * FWHM around the dip,
    clipped halfway to its neighbours. min_depth_db defaults to 6x the robust
    noise level of the flattened trace, so it adapts to each measurement.
    """
    freq = np.asarray(freq, dtype=float)
    mag = np.asarray(mag_db, dtype=float)
    n = len(mag)
    if n < 20:
        return {"noise_db": None, "min_depth_db": min_depth_db, "baseline_points": None, "resonances": []}
    df = (freq[-1] - freq[0]) / (n - 1)

    # Robust noise estimate from point-to-point differences (independent of the baseline).
    d = np.diff(mag)
    noise = 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)
    auto_depth = min_depth_db is None
    if auto_depth:
        min_depth_db = max(6 * noise, 0.3)

    win = int(baseline_points) if baseline_points else max(31, n // 100)
    for _ in range(4):
        win |= 1
        depth = _baseline(mag, win) - mag  # dips become positive peaks
        smooth = np.convolve(depth, np.ones(3) / 3, mode="same")
        peaks, props = find_peaks(smooth, prominence=min_depth_db, distance=3)
        if len(peaks) == 0:
            if baseline_points or win >= n // 3:
                break
            # Nothing found: maybe a zoomed sweep where one dip fills the span and the
            # running median follows it. Retry with a straight baseline through the edges.
            win = n
            continue
        # Width at half the dip in linear power: the FWHM of a Lorentzian notch.
        lin = 1 - 10 ** (-np.clip(smooth, 0, None) / 10)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # "some peaks have a width of 0" on 1-point spikes
            widths = np.maximum(peak_widths(lin, peaks, rel_height=0.5)[0], 3)
        if baseline_points or win >= n // 3 or 10 * widths.max() < win:
            break
        win = int(20 * widths.max())  # a dip wider than the median window gets eaten: grow it

    if len(peaks) == 0:
        return {"noise_db": float(noise), "min_depth_db": float(min_depth_db),
                "baseline_points": int(win), "resonances": []}

    keep = np.sort(np.argsort(props["prominences"])[::-1][:max_count])
    peaks, widths, prom = peaks[keep], widths[keep], props["prominences"][keep]

    out = []
    for i, p in enumerate(peaks):
        half = window_factor * widths[i]
        lo, hi = p - half, p + half
        if i > 0:
            lo = max(lo, (peaks[i - 1] + p) / 2)
        if i < len(peaks) - 1:
            hi = min(hi, (peaks[i + 1] + p) / 2)
        lo = int(max(0, np.floor(lo)))
        hi = int(min(n - 1, np.ceil(hi)))
        out.append({
            "f0": float(freq[p]),
            "depth_db": float(prom[i]),
            "fwhm": float(widths[i] * df),
            "f1": float(freq[lo]),
            "f2": float(freq[hi]),
        })
    return {"noise_db": float(noise), "min_depth_db": float(min_depth_db),
            "baseline_points": int(win), "resonances": out}


def recenter_window(freq, mag_db, f1, f2):
    """Shift a window [f1, f2] so it is centred on the deepest point inside it.

    Used to reuse one run's windows on other runs where resonances drift a bit.
    """
    freq = np.asarray(freq)
    sel = (freq >= f1) & (freq <= f2)
    if sel.sum() < 3:
        return f1, f2
    fmin = freq[sel][np.argmin(np.asarray(mag_db)[sel])]
    half = (f2 - f1) / 2
    return fmin - half, fmin + half
