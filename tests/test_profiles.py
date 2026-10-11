"""Tests: profiles, rotation curves, comparison harness, positivity."""

import numpy as np
import pytest

from tapered_sinc import (
    TaperedSincProfile, NFWProfile, BurkertProfile,
    total_velocity, chi2, compare, render_table,
    positivity_audit, audit_parameter_grid,
)
from tapered_sinc.fitting import make_model, DEFAULT_BOUNDS


def dwarf():
    return TaperedSincProfile(log10_rho0=7.5, rc=4.0, log10_rhos=6.8,
                              rs=12.0, log10_A=0.0, eta=0.1, truncate=True)


def test_core_density_finite_at_center():
    p = dwarf()
    # w(0) = 1/(1+e^-10) ≈ 0.99995, so density(0) ≈ rho0 up to taper edge
    assert p.density(0.0) == pytest.approx(p.rho0, rel=1e-3)
    assert np.isfinite(p.density(0.0))


def test_truncation_kills_negative_lobe():
    p = TaperedSincProfile(log10_rho0=7.5, rc=4.0, log10_rhos=6.8,
                           rs=12.0, log10_A=0.0, eta=0.1, truncate=True)
    r = np.linspace(0, 200, 2000)
    assert np.all(p.density(r) >= 0)


def test_untruncated_can_go_negative():
    # Documents WHY truncate=True is the default: without it, the taper
    # alone does not guarantee positivity for all parameter choices.
    p = TaperedSincProfile(log10_rho0=9.0, rc=1.0, log10_rhos=5.0,
                           rs=50.0, log10_A=-1.5, eta=0.3, truncate=False)
    ok, _, _ = p.check_positive()
    assert not ok  # expected failure captured as a test


def test_positivity_audit_grid():
    def make(params):
        lr0, rc, lrs, rs, lA = params
        return TaperedSincProfile(lr0, rc, lrs, rs, log10_A=lA,
                                  truncate=True)
    grid = [(7.5, 4.0, 6.8, 12.0, 0.0),
            (8.0, 2.0, 7.0, 8.0, 0.3),
            (7.0, 10.0, 6.5, 20.0, -0.3)]
    rep = audit_parameter_grid(make, grid)
    assert rep["n_failed"] == 0


def test_enclosed_mass_monotone():
    p = dwarf()
    r = np.linspace(0.1, 50, 100)
    m = p.enclosed_mass(r)
    assert np.all(np.diff(m) > 0)


def test_circular_velocity_shape():
    p = dwarf()
    r = np.linspace(0.5, 30, 60)
    v = p.circular_velocity(r)
    assert np.all(np.isfinite(v)) and np.all(v > 0)
    # cored profile: velocity rises roughly linearly at small r
    assert v[5] > v[0]


def test_nfw_burkert_sane():
    r = np.linspace(0.5, 30, 60)
    for prof in (NFWProfile(6.8, 12.0), BurkertProfile(7.5, 4.0)):
        v = prof.circular_velocity(r)
        assert np.all(np.isfinite(v)) and np.all(v > 0)


def test_total_velocity_baryons():
    p = dwarf()
    r = np.linspace(1, 10, 10)
    v_dm = total_velocity(r, p)
    v_with = total_velocity(r, p, v_disk=20 * np.ones(10),
                            v_gas=5 * np.ones(10), log10_ystar=0.0)
    assert np.all(v_with >= v_dm)


def test_chi2_perfect_fit_zero():
    v = np.array([10.0, 20.0, 30.0])
    assert chi2(v, v, np.ones(3)) == pytest.approx(0.0)


def test_compare_ranks_correctly():
    rows = compare([
        {"name": "a", "chi2": 10.0, "k": 6, "n": 20},
        {"name": "b", "chi2": 12.0, "k": 3, "n": 20},
    ])
    # b: AIC = 6+12 = 18 ; a: AIC = 12+10 = 22 -> b wins despite worse chi2
    assert rows[0]["name"] == "b"
    assert rows[0]["delta_aic"] == pytest.approx(0.0)


def test_compare_warns_on_bad_absolute_chi2():
    rows = compare([
        {"name": "a", "chi2": 400.0, "k": 6, "n": 73},
        {"name": "b", "chi2": 640.0, "k": 3, "n": 73},
    ])
    text = render_table(rows)
    assert "WARNING" in text


def test_make_model_finite():
    r = np.linspace(0.5, 10, 10)
    theta = [7.5, 4.0, 6.8, 12.0, 0.0, np.log10(0.5)]
    v = make_model(theta, r, v_disk=10 * np.ones(10))
    assert np.all(np.isfinite(v)) and np.all(v > 0)


def test_default_bounds_cover_pilot():
    # pilot medians from the paper should sit inside the prior box
    lows = np.array([b[0] for b in DEFAULT_BOUNDS])
    highs = np.array([b[1] for b in DEFAULT_BOUNDS])
    pilot = np.array([7.5, 4.31, 6.8, 12.0, 0.0, np.log10(0.5)])  # DDO154-ish
    assert np.all(pilot >= lows) and np.all(pilot <= highs)
