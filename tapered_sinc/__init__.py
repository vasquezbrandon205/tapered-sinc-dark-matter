"""Tapered-sinc dark matter halo model: profiles, fitting, comparison."""

from tapered_sinc.profiles import TaperedSincProfile, NFWProfile, BurkertProfile
from tapered_sinc.rotation_curve import total_velocity, chi2, residuals
from tapered_sinc.comparison import compare, render_table, aic, bic
from tapered_sinc.validation import (
    residual_diagnostics, positivity_audit, audit_parameter_grid,
    held_out_chi2,
)

__version__ = "0.1.0"
__all__ = [
    "TaperedSincProfile", "NFWProfile", "BurkertProfile",
    "total_velocity", "chi2", "residuals",
    "compare", "render_table", "aic", "bic",
    "residual_diagnostics", "positivity_audit", "audit_parameter_grid",
    "held_out_chi2",
]
