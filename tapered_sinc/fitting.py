"""
Fitting: least-squares initialization + MCMC sampling interface.

Stage 1: scipy least-squares for a stable starting point.
Stage 2: emcee MCMC around the LSQ solution for posteriors.
(emcee is imported lazily so the package works without it.)

Parameter vector for TaperedSincProfile:
    [log10_rho0, rc, log10_rhos, rs, log10_A, log10_ystar]
(eta fixed; truncate fixed True — see profiles.py)
"""

import numpy as np

from tapered_sinc.profiles import TaperedSincProfile
from tapered_sinc.rotation_curve import chi2, total_velocity


def make_model(theta, r, v_disk=None, v_bulge=None, v_gas=None, eta=0.1):
    log10_rho0, rc, log10_rhos, rs, log10_A, log10_ystar = theta
    halo = TaperedSincProfile(log10_rho0, rc, log10_rhos, rs,
                              log10_A=log10_A, eta=eta, truncate=True)
    return total_velocity(r, halo, v_disk, v_bulge, v_gas, log10_ystar)


def neg_loglike(theta, r, v_obs, v_err, v_disk=None, v_bulge=None,
                v_gas=None, eta=0.1):
    try:
        v_model = make_model(theta, r, v_disk, v_bulge, v_gas, eta)
    except ValueError:
        return np.inf
    if not np.all(np.isfinite(v_model)):
        return np.inf
    return 0.5 * chi2(v_model, v_obs, v_err)


def _scipy_bounds(bounds):
    """Convert [(lo, hi), ...] pairs to scipy's (lower, upper) format."""
    bounds = list(bounds)
    lower = [b[0] for b in bounds]
    upper = [b[1] for b in bounds]
    return (lower, upper)


def least_squares_fit(r, v_obs, v_err, theta0, bounds, v_disk=None,
                      v_bulge=None, v_gas=None, eta=0.1):
    """Stage 1: scipy least-squares initialization."""
    from scipy.optimize import least_squares

    def resid(theta):
        v_model = make_model(theta, r, v_disk, v_bulge, v_gas, eta)
        return (v_model - v_obs) / v_err

    sol = least_squares(resid, theta0, bounds=_scipy_bounds(bounds),
                        method="trf", max_nfev=20000)
    return sol.x, float(np.sum(sol.fun ** 2))


def mcmc_sample(r, v_obs, v_err, theta0, bounds, n_walkers=32,
                n_steps=5000, v_disk=None, v_bulge=None, v_gas=None,
                eta=0.1, seed=0):
    """Stage 2: emcee sampling around the LSQ solution.

    Returns the sampler. Caller handles burn-in / thinning / autocorrelation
    (require N_steps >= 50 * tau).
    """
    try:
        import emcee
    except ImportError as e:
        raise ImportError(
            "emcee is required for MCMC sampling: pip install emcee") from e

    rng = np.random.default_rng(seed)
    ndim = len(theta0)
    # uniform priors within bounds
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])

    def log_prob(theta):
        if np.any(theta < lower) or np.any(theta > upper):
            return -np.inf
        return -neg_loglike(theta, r, v_obs, v_err, v_disk, v_bulge,
                            v_gas, eta)

    p0 = theta0 + 1e-3 * rng.standard_normal((n_walkers, ndim))
    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_prob)
    sampler.run_mcmc(p0, n_steps, progress=False)
    return sampler


# Recommended broad, uninformative priors (log-space where noted):
#   log10_rho0 in [3, 13], rc in [0.01, 200] kpc,
#   log10_rhos in [3, 13], rs in [0.1, 2000] kpc,
#   log10_A in [-2, 2], log10_ystar in [log10(0.01), log10(10)]
DEFAULT_BOUNDS = [
    (3.0, 13.0),                          # log10_rho0
    (0.01, 200.0),                        # rc
    (3.0, 13.0),                          # log10_rhos
    (0.1, 2000.0),                        # rs
    (-2.0, 2.0),                          # log10_A
    (np.log10(0.01), np.log10(10.0)),     # log10_ystar
]
