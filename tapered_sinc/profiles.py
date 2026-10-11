"""
Tapered-sinc dark matter halo profiles.

Profiles:
    TaperedSincProfile — sinc core + logistic taper to NFW-like outer halo.
        rho(r) = w(r) * rho_core(r) + (1 - w(r)) * A * rho_outer(r)
        rho_core(r) = rho0 * sinc(pi * r / rc)
        w(r) = 1 / (1 + exp((r - rc) / Delta)), Delta = eta * rc
    NFWProfile         — Navarro-Frenk-White (comparison baseline)
    BurkertProfile     — Burkert cored profile (comparison baseline)

Physical validity:
    sinc(x) oscillates and goes negative past its first zero (x = pi).
    A physical density must be non-negative everywhere. Two mechanisms:
      1. The taper suppresses the core term before the negative lobe.
      2. Optional hard truncation: core term is exactly zero for r > rc.
    Use `check_positive()` to audit any parameter set, and
    `positivity_report()` for a grid scan. If the profile goes negative
    anywhere, the fit is unphysical — fix the model, not the plot.
"""

import numpy as np

# G in kpc (km/s)^2 / Msun
G_KPC = 4.3009e-6


def _sinc(x):
    """sin(x)/x, regularized at x=0."""
    return np.sinc(np.asarray(x) / np.pi)


class TaperedSincProfile:
    """Tapered-sinc halo: finite core, smooth taper to NFW-like outer."""

    PARAM_NAMES = ["log10_rho0", "rc", "log10_rhos", "rs", "log10_A",
                   "eta", "truncate"]

    def __init__(self, log10_rho0, rc, log10_rhos, rs, log10_A=0.0,
                 eta=0.1, truncate=True):
        """
        Args:
            log10_rho0: log10 central density [Msun/kpc^3]
            rc: core radius [kpc]
            log10_rhos: log10 NFW scale density [Msun/kpc^3]
            rs: NFW scale radius [kpc]
            log10_A: log10 outer amplitude (expect ~0 if taper is honest)
            eta: taper width as fraction of rc (Delta = eta * rc)
            truncate: if True, core term is exactly 0 for r > rc
                      (kills the negative sinc lobe by construction)
        """
        self.rho0 = 10.0 ** log10_rho0
        self.rc = float(rc)
        self.rhos = 10.0 ** log10_rhos
        self.rs = float(rs)
        self.A = 10.0 ** log10_A
        self.eta = float(eta)
        self.truncate = bool(truncate)
        if self.rc <= 0 or self.rs <= 0:
            raise ValueError("rc and rs must be positive")
        if self.eta <= 0:
            raise ValueError("eta must be positive")

    def _w(self, r):
        delta = self.eta * self.rc
        with np.errstate(over="ignore"):
            return 1.0 / (1.0 + np.exp((np.asarray(r) - self.rc) / delta))

    def rho_core(self, r):
        r = np.asarray(r, dtype=float)
        core = self.rho0 * _sinc(np.pi * r / self.rc)
        if self.truncate:
            core = np.where(r <= self.rc, core, 0.0)
        return core

    def rho_outer(self, r):
        r = np.asarray(r, dtype=float)
        x = r / self.rs
        # regularized at r=0 (outer term is suppressed by taper there anyway)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = self.rhos / (x * (1.0 + x) ** 2)
        return np.where(x > 0, out, self.rhos)

    def density(self, r):
        r = np.asarray(r, dtype=float)
        w = self._w(r)
        return w * self.rho_core(r) + (1.0 - w) * self.A * self.rho_outer(r)

    def check_positive(self, r_max=None, n=2000):
        """Audit: is rho(r) >= 0 on [0, r_max]? Returns (ok, min_rho, r_at_min)."""
        r_max = r_max or 20.0 * max(self.rc, self.rs)
        r = np.linspace(0, r_max, n)
        rho = self.density(r)
        i = int(np.argmin(rho))
        return bool(rho[i] >= 0), float(rho[i]), float(r[i])

    def enclosed_mass(self, r, n=2000):
        """M(<r) via trapezoidal integration. [Msun]"""
        r = np.atleast_1d(np.asarray(r, dtype=float))
        rmax = float(np.max(r))
        grid = np.linspace(0, rmax, n)
        rho = self.density(grid)
        integrand = 4.0 * np.pi * grid ** 2 * rho
        cum = np.concatenate([[0.0], np.cumtrapz(integrand, grid)])
        return np.interp(r, grid, cum)

    def circular_velocity(self, r):
        """Vc(r) = sqrt(G M(<r) / r). [km/s]"""
        r = np.atleast_1d(np.asarray(r, dtype=float))
        m = self.enclosed_mass(r)
        with np.errstate(divide="ignore", invalid="ignore"):
            v = np.sqrt(G_KPC * m / r)
        return np.where(r > 0, v, 0.0)


class NFWProfile:
    PARAM_NAMES = ["log10_rhos", "rs"]

    def __init__(self, log10_rhos, rs):
        self.rhos = 10.0 ** log10_rhos
        self.rs = float(rs)

    def density(self, r):
        r = np.asarray(r, dtype=float)
        x = r / self.rs
        with np.errstate(divide="ignore", invalid="ignore"):
            out = self.rhos / (x * (1.0 + x) ** 2)
        return np.where(x > 0, out, self.rhos)

    def enclosed_mass(self, r, n=2000):
        # analytic NFW enclosed mass
        r = np.atleast_1d(np.asarray(r, dtype=float))
        x = r / self.rs
        return 4 * np.pi * self.rhos * self.rs ** 3 * (
            np.log(1 + x) - x / (1 + x))

    def circular_velocity(self, r):
        r = np.atleast_1d(np.asarray(r, dtype=float))
        m = self.enclosed_mass(r)
        with np.errstate(divide="ignore", invalid="ignore"):
            v = np.sqrt(G_KPC * m / r)
        return np.where(r > 0, v, 0.0)

    def check_positive(self, r_max=None, n=2000):
        return True, 0.0, 0.0


class BurkertProfile:
    PARAM_NAMES = ["log10_rho0", "rc"]

    def __init__(self, log10_rho0, rc):
        self.rho0 = 10.0 ** log10_rho0
        self.rc = float(rc)

    def density(self, r):
        r = np.asarray(r, dtype=float)
        x = r / self.rc
        return self.rho0 / ((1 + x) * (1 + x ** 2))

    def enclosed_mass(self, r, n=2000):
        r = np.atleast_1d(np.asarray(r, dtype=float))
        x = r / self.rc
        return (2 * np.pi * self.rho0 * self.rc ** 3 *
                (np.log((1 + x) ** 2 * (1 + x ** 2)) - 2 * np.arctan(x)))

    def circular_velocity(self, r):
        r = np.atleast_1d(np.asarray(r, dtype=float))
        m = self.enclosed_mass(r)
        with np.errstate(divide="ignore", invalid="ignore"):
            v = np.sqrt(G_KPC * m / r)
        return np.where(r > 0, v, 0.0)

    def check_positive(self, r_max=None, n=2000):
        return True, 0.0, 0.0


# np.cumtrapz was removed in numpy 2.0; provide fallback
if not hasattr(np, "cumtrapz"):
    def _cumtrapz(y, x):
        dx = np.diff(x)
        return np.cumsum(0.5 * (y[1:] + y[:-1]) * dx)
    np.cumtrapz = _cumtrapz
