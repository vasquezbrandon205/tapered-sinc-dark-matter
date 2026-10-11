"""
Positivity audit demo: the single most important check in this repo.

sinc(x) goes negative past x=pi. This script:
  1. Shows the raw (untruncated) core going negative past rc.
  2. Shows the taper suppressing it — and where that breaks down.
  3. Shows hard truncation (truncate=True) killing the lobe by construction.

Run: python examples/positivity_audit.py
"""

import numpy as np

from tapered_sinc import TaperedSincProfile, positivity_audit


def main():
    base = dict(log10_rho0=7.5, rc=4.0, log10_rhos=6.8, rs=12.0,
                log10_A=0.0, eta=0.1)

    print("Case 1: truncate=False (rely on taper alone)")
    p1 = TaperedSincProfile(truncate=False, **base)
    a1 = positivity_audit(p1)
    print(f"  ok={a1['ok']}, min_rho={a1['min_rho']:.3e} "
          f"at r={a1['r_at_min_kpc']:.2f} kpc")

    print("Case 2: truncate=True (core term exactly 0 for r > rc)")
    p2 = TaperedSincProfile(truncate=True, **base)
    a2 = positivity_audit(p2)
    print(f"  ok={a2['ok']}, min_rho={a2['min_rho']:.3e} "
          f"at r={a2['r_at_min_kpc']:.2f} kpc")

    print("Case 3: wide taper (eta=0.5), truncate=False — stress test")
    base3 = dict(base); base3["eta"] = 0.5
    p3 = TaperedSincProfile(truncate=False, **base3)
    a3 = positivity_audit(p3)
    print(f"  ok={a3['ok']}, min_rho={a3['min_rho']:.3e} "
          f"at r={a3['r_at_min_kpc']:.2f} kpc")

    print()
    if not a1["ok"]:
        print("Case 1 FAILS: taper alone does not guarantee positivity.")
    if a2["ok"]:
        print("Case 2 PASSES: hard truncation is the safe default.")
    if not a3["ok"]:
        print("Case 3 FAILS: wide tapers leak the negative lobe — "
              "another reason truncate=True is default.")


if __name__ == "__main__":
    main()
