"""Plot the synthetic resonances used by the tests, with their fits.

    python tests/plot_synthetic.py            # -> tests/synthetic_fits.png

Each row is one test case at SNR 30 dB: |S| (dB), unwrapped phase and the IQ plane,
data in blue and the fitted model in orange, with true vs fitted parameters.
"""
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from fitting import fit_resonance, to_complex  # noqa: E402
from synthetic import Environment, Resonance, trace, window  # noqa: E402

warnings.filterwarnings("ignore")

DATA, FIT, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e7e6e2"

CASES = [
    ("Critically coupled notch", Resonance(fr=5.0e9, Qi=1e5, Qc=1e5)),
    ("Over-coupled ×10 notch  (Qc < Qi: deep dip)", Resonance(fr=5.0e9, Qi=1e6, Qc=1e5)),
    ("Under-coupled ×10 notch  (Qi < Qc: shallow dip)", Resonance(fr=5.0e9, Qi=1e4, Qc=1e5)),
    ("Asymmetric notch, φ = +0.4", Resonance(fr=5.0e9, Qi=2e5, Qc=1e5, phi=0.4)),
    ("Low-Q notch", Resonance(fr=4.0e9, Qi=3e3, Qc=2e3)),
    ("Over-coupled reflection", Resonance(fr=6.0e9, Qi=2e5, Qc=2e4, port="reflection")),
]


def si(v):
    a = abs(v)
    return f"{v / 1e6:.3g}M" if a >= 1e6 else f"{v / 1e3:.3g}k" if a >= 1e3 else f"{v:.3g}"


def main(out=ROOT / "tests" / "synthetic_fits.png", snr_db=30, seed=7):
    rng = np.random.default_rng(seed)
    env = Environment(snr_db=snr_db)       # a = -40 dB, alpha = 0.7 rad, 40 ns cable delay
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
                         "grid.color": GRID, "grid.linewidth": 0.6})
    fig, axes = plt.subplots(len(CASES), 3, figsize=(15.5, 3.0 * len(CASES)),
                             gridspec_kw={"width_ratios": [1.25, 1.25, 1]})
    for row, (title, res) in enumerate(CASES):
        f = window(res)
        mag, ph = trace(f, [res], env, rng)
        o = fit_resonance(f, mag, ph, f[0], f[-1], port=res.port)
        x = (f - res.fr) / 1e3                                   # kHz from the true fr
        mf = (np.array(o["model"]["freq"]) - res.fr) / 1e3
        a_mag, a_ph, a_iq = axes[row]

        a_mag.plot(x, mag, color=DATA, lw=0.8, label="synthetic data")
        a_mag.plot(mf, o["model"]["mag"], color=FIT, lw=2, label="fit")
        a_mag.set_ylabel("|S| (dB)")
        # Remove the known 40 ns cable delay so the resonance feature is visible.
        ramp = lambda ff: 360 * env.delay * (ff - res.fr)
        a_ph.plot(x, ph + ramp(f), color=DATA, lw=0.8)
        a_ph.plot(mf, np.array(o["model"]["phase"]) + ramp(np.array(o["model"]["freq"])), color=FIT, lw=2)
        a_ph.set_ylabel("phase − delay (°)")
        for a in (a_mag, a_ph):
            a.axvline((o["fr"] - res.fr) / 1e3, color=MUTED, lw=0.6, ls=":")
            a.set_xlabel("f − fr (kHz)")

        z = to_complex(mag, ph)
        zm = to_complex(np.array(o["model"]["mag"]), np.array(o["model"]["phase"]))
        a_iq.plot(z.real * 1e3, z.imag * 1e3, ".", color=DATA, ms=1.5)
        a_iq.plot(zm.real * 1e3, zm.imag * 1e3, color=FIT, lw=2)
        a_iq.set_aspect("equal", "datalim")
        a_iq.set_xlabel("Re S (×10⁻³)")
        a_iq.set_ylabel("Im S (×10⁻³)")

        def cmp(k):
            t, v, e = getattr(res, k), o[k], o[k + "_err"]
            return f"{k} {si(t)} → {si(v)} ± {si(e)}  ({(v - t) / e:+.1f}σ)"
        a_mag.set_title(title, loc="left", fontsize=10, color=INK, fontweight="bold")
        a_ph.set_title("true → fitted:  " + "   ".join(cmp(k) for k in ("Qi", "Qc")),
                       loc="left", fontsize=8.5, color=INK)
        a_iq.set_title(f"{cmp('QL')}\nfr off {abs(o['fr'] - res.fr):.0f} Hz · delay {o['delay'] * 1e9:.2f} ns"
                       f" (true 40) · res/noise {o['resid_noise']:.2f}", loc="left", fontsize=8.5, color=INK)
        if row == 0:
            a_mag.legend(frameon=False, loc="lower left")
    fig.suptitle(f"Synthetic resonances (SNR {snr_db} dB, 40 ns cable delay) and their fits",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out, dpi=110)
    print(out)


if __name__ == "__main__":
    main()
