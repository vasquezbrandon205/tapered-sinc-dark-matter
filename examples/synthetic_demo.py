"""
Synthetic demo: generate a rotation curve from a tapered-sinc halo,
fit it back with least squares, and compare against NFW/Burkert.

No SPARC data needed — validates that the pipeline recovers known
parameters and that the comparison harness works.

Run: python examples/synthetic_demo.py
"""

import numpy as np

from tapered_sinc import (
    TaperedSincProfile, NFWProfile, BurkertProfile,
    total_velocity, compare, render_table, positivity_audit,
)
from tapered_sinc.fitting import least_squares_fit, DEFAULT_BOUNDS


def main():
    rng = np.random.default_rng(42)

    # True halo: a dwarf-like tapered-sinc profile
    true = TaperedSincProfile(
        log10_rho0=7.5, rc=4.0,
        log10_rhos=6.8, rs=12.0,
        log10_A=0.0, eta=0.1, truncate=True,
    )
    audit = positivity_audit(true)
    print(f"positivity audit (true profile): ok={audit['ok']}, "
          f"min_rho={audit['min_rho']:.3e}")
    assert audit["ok"], "true profile fails positivity — fix the model!"

    r = np.linspace(0.5, 15.0, 25)
    v_disk = 20.0 * np.exp(-r / 3.0)          # toy stellar disk
    v_gas = 8.0 * np.ones_like(r)             # toy gas
    v_true = total_velocity(r, true, v_disk=v_disk, v_gas=v_gas,
                            log10_ystar=np.log10(0.5))
    v_err = 2.0 * np.ones_like(r)
    v_obs = v_true + rng.normal(0, v_err)

    # Fit tapered-sinc (LSQ stage)
    theta0 = np.array([7.0, 3.0, 6.5, 10.0, 0.0, np.log10(0.4)])
    theta_hat, c2 = least_squares_fit(
        r, v_obs, v_err, theta0, DEFAULT_BOUNDS,
        v_disk=v_disk, v_gas=v_gas)
    print(f"\ntapered-sinc LSQ: chi2={c2:.1f}, "
          f"rc_hat={theta_hat[1]:.2f} (true 4.00)")

    # Fit NFW and Burkert for comparison (fewer params)


    def fit_simple(make_halo, t0, lo, hi):
        from scipy.optimize import least_squares as lsq

        def resid(t):
            halo = make_halo(*t[:-1])
            vm = total_velocity(r, halo, v_disk=v_disk, v_gas=v_gas,
                                log10_ystar=t[-1])
            return (vm - v_obs) / v_err
        sol = lsq(resid, t0, bounds=(lo, hi), method="trf", max_nfev=20000)
        return float(np.sum(sol.fun ** 2))

    c2_nfw = fit_simple(
        lambda a, b: NFWProfile(a, b), [6.5, 10.0, np.log10(0.4)],
        [3, 0.1, np.log10(0.01)], [13, 2000, np.log10(10)])
    c2_bur = fit_simple(
        lambda a, b: BurkertProfile(a, b), [7.0, 3.0, np.log10(0.4)],
        [3, 0.01, np.log10(0.01)], [13, 200, np.log10(10)])

    rows = compare([
        {"name": "tapered-sinc", "chi2": c2, "k": 6, "n": len(r)},
        {"name": "NFW", "chi2": c2_nfw, "k": 3, "n": len(r)},
        {"name": "Burkert", "chi2": c2_bur, "k": 3, "n": len(r)},
    ])
    print("\n" + render_table(rows))
    print("\nTrue model was tapered-sinc; it should win on AIC here.")


if __name__ == "__main__":
    main()
