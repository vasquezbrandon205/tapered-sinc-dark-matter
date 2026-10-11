"""
Validation: residuals, held-out evaluation, and the positivity audit.

The positivity audit is the single most important check in this repo:
sinc oscillates and goes negative past its first zero. A density that
goes negative anywhere is unphysical and invalidates the fit — no matter
how good the chi2 looks.
"""

import numpy as np

from tapered_sinc.rotation_curve import residuals


def residual_diagnostics(v_model, v_obs, v_err):
    """Standardized residuals + simple normality-ish summaries."""
    r = residuals(v_model, v_obs, v_err)
    return {
        "residuals": r,
        "mean": float(np.mean(r)),
        "std": float(np.std(r)),
        "max_abs": float(np.max(np.abs(r))),
        "frac_within_1sigma": float(np.mean(np.abs(r) < 1.0)),
        "frac_within_2sigma": float(np.mean(np.abs(r) < 2.0)),
    }


def positivity_audit(profile, r_max_kpc=500.0, n=5000):
    """Scan rho(r) on [0, r_max] — fail loudly if negative anywhere.

    Returns dict with ok flag, minimum density, and its location.
    """
    ok, min_rho, r_at_min = profile.check_positive(r_max_kpc, n)
    return {
        "ok": ok,
        "min_rho": min_rho,
        "r_at_min_kpc": r_at_min,
        "r_max_kpc": r_max_kpc,
    }


def audit_parameter_grid(make_profile, param_grid, r_max_kpc=500.0):
    """Run the positivity audit over many parameter sets (e.g. MCMC draws).

    Args:
        make_profile: callable(params) -> profile
        param_grid: iterable of parameter tuples

    Returns:
        dict with n_total, n_failed, and the worst offending params.
    """
    failed = []
    worst = (float("inf"), None)
    n = 0
    for params in param_grid:
        n += 1
        prof = make_profile(params)
        ok, min_rho, r_at = prof.check_positive(r_max_kpc)
        if not ok:
            failed.append((tuple(params), min_rho, r_at))
        if min_rho < worst[0]:
            worst = (min_rho, tuple(params))
    return {
        "n_total": n,
        "n_failed": len(failed),
        "failures": failed[:10],
        "worst_min_rho": worst[0],
        "worst_params": worst[1],
    }


def held_out_chi2(v_model_heldout, v_obs_heldout, v_err_heldout):
    """Chi2 on galaxies / radii not used in fitting. The honest metric."""
    from tapered_sinc.rotation_curve import chi2
    return chi2(v_model_heldout, v_obs_heldout, v_err_heldout)
