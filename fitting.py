"""Resonance detection and notch-port circle fits (resonator_tools, as in RFP-Software)."""
import warnings

import matplotlib

matplotlib.use("Agg")  # resonator_tools imports pyplot; keep it headless

import numpy as np
from scipy.ndimage import median_filter
from scipy.optimize import least_squares
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


def _shape(port, f, fr, QL, Qc, phi):
    """Ideal resonator response (no environment), resonator_tools conventions."""
    if port == "reflection":
        x = 2j * QL * (fr - f) / fr
        return (2 * QL / Qc - 1 + x) / (1 - x)
    return 1 - (QL / Qc) * np.exp(1j * phi) / (1 + 2j * QL * (f - fr) / fr)


def _refine(port, f, z, fr, QL, Qc, phi, delay, fix_delay):
    """Full complex least-squares fit of the whole model, started from the circle fit.

    Fits fr, QL, |Qc|, phi (notch), cable delay (unless fixed) and the complex
    amplitude a*e^{i alpha} together. The circle fit determines the delay, circle and
    phase in separate steps, so a delay error leaks straight into Q (worst for
    symmetric, phi ~ 0 resonances); fitting everything at once removes that and the
    covariance gives honest error bars. Returns None if it fails.
    """
    f0 = f.mean()
    span2pi = 2 * np.pi * max(f[-1] - f[0], 1.0)
    w0 = fr / QL
    g0 = _shape(port, f, fr, QL, Qc, phi) * np.exp(-2j * np.pi * delay * (f - f0))
    A0 = np.vdot(g0, z) / np.vdot(g0, g0)      # best complex amplitude for the start values
    scale = abs(A0)
    notch = port != "reflection"

    def unpack(u):
        k = 0
        frr = fr + w0 * u[k]; k += 1
        ql = np.exp(u[k]); k += 1
        qc = np.exp(u[k]); k += 1
        ph = 0.0
        if notch:
            ph = u[k]; k += 1
        tau = delay
        if not fix_delay:
            tau = u[k] / span2pi; k += 1
        A = scale * (u[k] + 1j * u[k + 1])
        return frr, ql, qc, ph, tau, A

    def model(u):
        frr, ql, qc, ph, tau, A = unpack(u)
        return A * np.exp(-2j * np.pi * tau * (f - f0)) * _shape(port, f, frr, ql, qc, ph)

    def resid(u):
        d = (model(u) - z) / scale
        return np.concatenate([d.real, d.imag])

    u0 = [0.0, np.log(QL), np.log(Qc)] + ([phi] if notch else []) + \
         ([] if fix_delay else [delay * span2pi]) + [A0.real / scale, A0.imag / scale]
    try:
        with np.errstate(all="ignore"):   # trial steps may pass through QL -> inf
            sol = least_squares(resid, u0, method="lm", x_scale=1.0, max_nfev=2000)
    except Exception:
        return None
    if not sol.success and sol.status <= 0:
        return None
    u = sol.x
    n, k = 2 * len(f), len(u)
    s2 = 2 * sol.cost / max(n - k, 1)
    try:
        cov = np.linalg.inv(sol.jac.T @ sol.jac) * s2
    except np.linalg.LinAlgError:
        cov = np.full((k, k), np.nan)
    frr, ql, qc, ph, tau, A = unpack(u)

    # Qi from 1/Qi = 1/QL - cos(phi)/|Qc| (notch, diameter-corrected) or 1/QL - 1/Qc.
    inv_qi = 1 / ql - (np.cos(ph) if notch else 1.0) / qc
    qi = 1 / inv_qi
    grad = np.zeros(k)             # d Qi / d u
    grad[1] = qi ** 2 / ql         # u1 = ln QL
    grad[2] = -qi ** 2 * (np.cos(ph) if notch else 1.0) / qc   # u2 = ln Qc
    if notch:
        grad[3] = -qi ** 2 * np.sin(ph) / qc
    var = lambda i: cov[i, i]
    zfit = model(u)
    return {
        "fr": frr, "fr_err": w0 * np.sqrt(var(0)),
        "QL": ql, "QL_err": ql * np.sqrt(var(1)),
        "Qc": qc, "Qc_err": qc * np.sqrt(var(2)),
        "Qi": qi, "Qi_err": np.sqrt(grad @ cov @ grad),
        "phi0": ph if notch else None,
        "delay": tau,
        "chi_square": s2,
        "z_model": zfit,
        "scale": scale,
    }


def _noise_var(z):
    """Per-quadrature noise variance from point-to-point differences (robust to the resonance)."""
    d = np.diff(z)
    mad = lambda x: 1.4826 * np.median(np.abs(x - np.median(x)))
    return (mad(d.real) ** 2 + mad(d.imag) ** 2) / 4    # diff doubles the variance; 2 quadratures


def _edge_delay(f, z, frac=0.15):
    """Cable delay from the phase slope of the off-resonance edges of the window."""
    k = max(3, int(len(f) * frac))
    idx = np.r_[np.arange(k), np.arange(len(f) - k, len(f))]
    ph = np.unwrap(np.angle(z))
    # Both edges are fitted with one slope and separate offsets (the resonance may add 2*pi).
    x = f[idx] - f.mean()
    A = np.column_stack([x, (idx < k).astype(float), (idx >= k).astype(float)])
    slope = np.linalg.lstsq(A, ph[idx], rcond=None)[0][0]
    return -slope / (2 * np.pi)


def _data_starts(port, f, z, delay):
    """Rough (fr, QL, Qc, phi) starting points read off the data, independent of the circle fit."""
    f0 = f.mean()
    zz = z * np.exp(2j * np.pi * delay * (f - f0))
    k = max(3, len(f) // 10)
    # Off-resonance baseline: line between the two edge means (complex).
    b0, b1 = zz[:k].mean(), zz[-k:].mean()
    base = b0 + (b1 - b0) * (f - f[:k].mean()) / (f[-k:].mean() - f[:k].mean())
    s = zz / base
    m = np.convolve(np.abs(s), np.ones(5) / 5, mode="same")
    m[:2], m[-2:] = 1, 1
    if port == "reflection":
        dph = np.abs(np.gradient(np.unwrap(np.angle(s)), f))
        i = int(np.argmax(np.convolve(dph, np.ones(5) / 5, mode="same")))
    else:
        i = int(np.argmin(m))
    fr = f[i]
    depth = float(np.clip(m[i], 0.0, 0.99))
    dip = 1 - m ** 2
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w = peak_widths(dip, [i], rel_height=0.5)[0][0] * (f[1] - f[0])
    except Exception:
        w = 0
    if not w > 0:
        w = (f[-1] - f[0]) / 16
    QL = fr / w
    starts = []
    for mult in (1.0, 0.5, 2.0, 0.25, 4.0):
        ql = QL * mult
        if port == "reflection":
            qc = 2 * ql / (1 + depth)        # over-coupled branch: |S11(fr)| = 2QL/Qc - 1
        else:
            qc = ql / max(1 - depth, 0.02)   # notch: |S21(fr)| = 1 - QL/Qc for phi = 0
        starts.append((fr, ql, qc, 0.0))
    return starts


def fit_resonance(freq, mag_db, phase_deg, f1, f2, port="notch", guessdelay=True,
                  electric_delay=None, refine=True, max_model_points=2000):
    """Fit the window [f1, f2] with resonator_tools (circuit.notch_port or reflection_port).

    Returns fr, QL, Qi, |Qc| with errors, plus the model curve (magnitude dB /
    phase deg) over the window. For notch, Qi is the diameter-corrected value
    (Qi_dia_corr), as in RFP-Software. With refine=True (default) the circle fit is
    only the starting point of a full-model least-squares fit (see _refine).
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
            # No linear-phase delay guess here: the 2*pi phase turn of a reflection
            # resonance throws it off (resonator_tools hardcodes guessdelay=False).
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

    ql, ql_err = r.get("Ql"), r.get("Ql_err")
    fr, fr_err, phi0, chi2 = r.get("fr"), r.get("fr_err"), r.get("phi0"), r.get("chi_square")
    z_model = p.z_data_sim
    method = "circle"
    if refine:
        fix = electric_delay is not None
        noise = _noise_var(z)
        q_window = np.mean(f) / ((f[-1] - f[0]) / 2)   # QL if the window were one linewidth wide

        def plausible(c):
            return (c is not None and c["QL"] > 0 and c["Qc"] > 0 and np.isfinite(c["Qi"])
                    and f[0] <= c["fr"] <= f[-1]
                    # linewidth narrower than the window (else it is just background) and not absurdly sharp
                    and 0.5 * q_window < c["QL"] < 1e4 * q_window)

        def good(c):
            # Residuals at the noise level (chi_square is per quadrature, in units of |a|^2).
            return plausible(c) and c["chi_square"] * c["scale"] ** 2 < 1.25 * noise

        cands = []
        if all(v is not None and np.isfinite(v) for v in (fr, ql, qc, delay)) and ql > 0 and qc > 0:
            cands.append(_refine(port, f, z, fr, ql, qc, phi0 if phi0 is not None else 0.0, delay, fix_delay=fix))
        if not cands or not good(cands[0]):
            # Circle fit missed (typical for strongly over-coupled or noisy resonances):
            # also start from values read off the data and keep the best fit.
            for d0 in ([electric_delay] if fix else {delay if delay is not None and np.isfinite(delay) else 0.0,
                                                      _edge_delay(f, z)}):
                for s0 in _data_starts(port, f, z, d0):
                    c = _refine(port, f, z, *s0, d0, fix_delay=fix)
                    cands.append(c)
                    if good(c):
                        break
                if any(good(c) for c in cands):
                    break
        ok = [c for c in cands if plausible(c)]
        ref = min(ok, key=lambda c: c["chi_square"] * c["scale"] ** 2) if ok else None
        if ref is not None:
            fr, fr_err, ql, ql_err = ref["fr"], ref["fr_err"], ref["QL"], ref["QL_err"]
            qi, qi_err, qc, qc_err = ref["Qi"], ref["Qi_err"], ref["Qc"], ref["Qc_err"]
            phi0, delay, chi2, z_model = ref["phi0"], ref["delay"], ref["chi_square"], ref["z_model"]
            method = "full model"

    # Goodness of fit: rms residual / noise level (~1 when the model describes the data).
    noise_rms = np.sqrt(_noise_var(z))
    resid_ratio = np.sqrt(np.mean(np.abs(z - z_model) ** 2) / 2) / noise_rms if noise_rms > 0 else np.nan

    # Keep the model curve light enough to plot.
    step = max(1, len(f) // max_model_points)
    zs = z_model[::step]
    model_phase = np.rad2deg(np.unwrap(np.angle(zs)))
    # Align the model's phase branch with the data's unwrapped phase.
    data_phase = np.asarray(phase_deg)[sel][::step]
    model_phase += 360 * np.round(np.median(data_phase - model_phase) / 360)

    flags = []
    if qi is not None and qi <= 0:
        flags.append("Qi ≤ 0 (unphysical; nonlinear or bad window?)")
    for name, v, e in (("Qi", qi, qi_err), ("Qc", qc, qc_err), ("QL", ql, ql_err)):
        if v is not None and e is not None and np.isfinite(e) and abs(e) > abs(v):
            flags.append(f"{name} error > 100%")
    if not (f[0] <= fr <= f[-1]):
        flags.append("fr outside window")
    if np.isfinite(resid_ratio) and resid_ratio > 3:
        flags.append(f"residuals {resid_ratio:.0f}× noise (line shape not Lorentzian: nonlinear/distorted?)")

    return {
        "flags": flags,
        "port": port,
        "method": method,
        "f1": float(f[0]),
        "f2": float(f[-1]),
        "n_points": int(len(f)),
        "fr": _finite(fr),
        "fr_err": _finite(fr_err),
        "QL": _finite(ql),
        "QL_err": _finite(ql_err),
        "Qi": _finite(qi),
        "Qi_err": _finite(qi_err),
        "Qc": _finite(qc),
        "Qc_err": _finite(qc_err),
        "phi0": _finite(phi0),
        "chi_square": _finite(chi2),
        "resid_noise": _finite(resid_ratio),
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
