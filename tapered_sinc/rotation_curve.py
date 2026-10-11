"""
Rotation curves: DM halo + baryons -> model circular velocity.

Standard SPARC treatment:
    V_model^2 = V_dm^2 + Ystar * V_disk^2 + Ystar * V_bulge^2 + V_gas^2

Ystar (stellar mass-to-light ratio) scales the stellar contributions;
gas is taken as given. All V components in km/s, r in kpc.
"""

import numpy as np


def total_velocity(r, halo, v_disk=None, v_bulge=None, v_gas=None,
                   log10_ystar=0.0):
    """Model circular velocity at radii r.

    Args:
        r: radii [kpc]
        halo: profile object with .circular_velocity(r)
        v_disk, v_bulge, v_gas: baryonic contributions [km/s] at r
            (arrays matching r, or None if absent)
        log10_ystar: log10 stellar mass-to-light ratio
    """
    r = np.asarray(r, dtype=float)
    ystar = 10.0 ** log10_ystar
    v2 = np.asarray(halo.circular_velocity(r), dtype=float) ** 2
    if v_disk is not None:
        v2 = v2 + ystar * np.asarray(v_disk, dtype=float) ** 2
    if v_bulge is not None:
        v2 = v2 + ystar * np.asarray(v_bulge, dtype=float) ** 2
    if v_gas is not None:
        v2 = v2 + np.asarray(v_gas, dtype=float) ** 2
    return np.sqrt(np.maximum(v2, 0.0))


def chi2(v_model, v_obs, v_err):
    """Chi-squared for independent Gaussian velocity errors."""
    v_model = np.asarray(v_model, dtype=float)
    v_obs = np.asarray(v_obs, dtype=float)
    v_err = np.asarray(v_err, dtype=float)
    return float(np.sum(((v_model - v_obs) / v_err) ** 2))


def residuals(v_model, v_obs, v_err):
    """Standardized residuals (should scatter ~ N(0,1) for a good fit)."""
    return (np.asarray(v_model) - np.asarray(v_obs)) / np.asarray(v_err)
