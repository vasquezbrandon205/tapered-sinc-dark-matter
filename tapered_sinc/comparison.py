"""
Model comparison: chi-squared, AIC, BIC on a consistent footing.

Rules:
  - Every model sees the same data, same error model, same Ystar treatment.
  - k = number of free parameters actually fit (not the size of the prior box).
  - AIC = 2k + chi2  (predictive accuracy)
  - BIC = k ln N + chi2  (stronger complexity penalty)
  - Report delta-AIC / delta-BIC vs the best model, not just raw values.
  - Absolute chi2/dof >> 1 for ALL models signals underestimated errors
    (known SPARC issue) — then only relative comparisons are meaningful.
"""

import numpy as np

from tapered_sinc.rotation_curve import chi2


def aic(chi2_val, k):
    return 2.0 * k + chi2_val


def bic(chi2_val, k, n):
    return k * np.log(n) + chi2_val


def compare(models):
    """Compare fitted models.

    Args:
        models: list of dicts with keys:
            name, chi2, k (free params), n (data points)

    Returns:
        list of dicts sorted by AIC, each augmented with aic, bic,
        delta_aic, delta_bic, chi2_dof.
    """
    rows = []
    for m in models:
        c2 = float(m["chi2"])
        k = int(m["k"])
        n = int(m["n"])
        rows.append({
            "name": m["name"],
            "chi2": c2,
            "k": k,
            "n": n,
            "chi2_dof": c2 / max(n - k, 1),
            "aic": aic(c2, k),
            "bic": bic(c2, k, n),
        })
    rows.sort(key=lambda r: r["aic"])
    best_aic = rows[0]["aic"]
    best_bic = min(r["bic"] for r in rows)
    for r in rows:
        r["delta_aic"] = r["aic"] - best_aic
        r["delta_bic"] = r["bic"] - best_bic
    return rows


def render_table(rows):
    lines = []
    lines.append(f"{'model':<18}{'chi2':>10}{'k':>4}{'chi2/dof':>10}"
                 f"{'AIC':>10}{'dAIC':>8}{'BIC':>10}{'dBIC':>8}")
    lines.append("-" * 78)
    for r in rows:
        lines.append(
            f"{r['name']:<18}{r['chi2']:>10.1f}{r['k']:>4}"
            f"{r['chi2_dof']:>10.2f}{r['aic']:>10.1f}{r['delta_aic']:>8.1f}"
            f"{r['bic']:>10.1f}{r['delta_bic']:>8.1f}")
    lines.append("")
    lines.append("dAIC/dBIC = 0 for the best model. Rule of thumb: dAIC > 10 "
                 "is strong evidence against.")
    all_bad = all(r["chi2_dof"] > 3.0 for r in rows)
    if all_bad:
        lines.append("WARNING: all models have chi2/dof >> 1 — errors likely "
                     "underestimated; trust relative rankings, not absolute chi2.")
    return "\n".join(lines)
