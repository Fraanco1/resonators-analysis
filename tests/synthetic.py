"""Synthetic resonances with known parameters, from the same model resonator_tools fits.

Notch (Probst et al., Rev. Sci. Instrum. 86, 024706 (2015)):

    S21(f) = a e^{i alpha} e^{-2 pi i f tau} [1 - (QL/|Qc|) e^{i phi} / (1 + 2i QL (f/fr - 1))]

with the diameter-corrected relation 1/QL = 1/Qi + cos(phi)/|Qc|. The app reports
Qi (= Qi_dia_corr) and Qc (= |Qc|), so those are the "truth" values here.

Reflection:

    S11(f) = a e^{i alpha} e^{-2 pi i f tau} (2 QL/Qc - 1 + 2i QL (fr - f)/fr) / (1 - 2i QL (fr - f)/fr)

with 1/QL = 1/Qi + 1/Qc.
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Resonance:
    fr: float = 5e9
    Qi: float = 2e5
    Qc: float = 1e5          # |Qc|
    phi: float = 0.0         # impedance-mismatch angle (notch only)
    port: str = "notch"

    @property
    def QL(self):
        if self.port == "reflection":
            return 1 / (1 / self.Qi + 1 / self.Qc)
        return 1 / (1 / self.Qi + np.cos(self.phi) / self.Qc)

    @property
    def fwhm(self):
        return self.fr / self.QL

    def s(self, f):
        """Ideal response (no environment)."""
        QL = self.QL
        if self.port == "reflection":
            x = 2j * QL * (self.fr - f) / self.fr
            return (2 * QL / self.Qc - 1 + x) / (1 - x)
        return 1 - (QL / self.Qc) * np.exp(1j * self.phi) / (1 + 2j * QL * (f / self.fr - 1))


@dataclass
class Environment:
    a: float = 0.01           # baseline amplitude (-40 dB)
    alpha: float = 0.7        # phase offset (rad)
    delay: float = 40e-9      # cable delay (s)
    snr_db: float = 40.0      # baseline amplitude / noise std per quadrature, in dB (None = no noise)

    def apply(self, f, z, rng):
        out = self.a * np.exp(1j * self.alpha) * np.exp(-2j * np.pi * f * self.delay) * z
        if self.snr_db is not None:
            sigma = self.a * 10 ** (-self.snr_db / 20)
            out = out + sigma * (rng.normal(size=f.size) + 1j * rng.normal(size=f.size))
        return out


def window(res, half_fwhm=8.0, points_per_fwhm=50):
    """Frequency grid of +-half_fwhm linewidths, like the auto-detect windows."""
    n = int(2 * half_fwhm * points_per_fwhm) + 1
    return np.linspace(res.fr - half_fwhm * res.fwhm, res.fr + half_fwhm * res.fwhm, n)


def trace(f, resonances, env, rng):
    """Magnitude (dB) and unwrapped phase (deg), as the database stores them."""
    z = np.ones_like(f, dtype=complex)
    for r in resonances:
        z = z * r.s(f)
    z = env.apply(f, z, rng)
    return 20 * np.log10(np.abs(z)), np.rad2deg(np.unwrap(np.angle(z)))
