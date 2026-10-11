# tapered-sinc-dark-matter

A unified dark-matter halo density profile — sinc core, smooth taper to an NFW-like outer halo — fit to galaxy rotation curves with full MCMC and honest model comparison.

```
rho(r) = w(r) * rho0 * sinc(pi*r/rc) + (1 - w(r)) * A * rho_NFW(r)
```

## Why this exists

Dwarf and low-surface-brightness galaxies show flat central density cores, in tension with the cuspy NFW profile from collisionless cold dark matter. This repo implements a profile with a finite sinc-like core (motivated by wave-like/dispersive DM dynamics) that tapers smoothly to an NFW outer halo — one functional form across dwarfs, LSBs, and spirals.

## The one thing you must know

`sinc(x)` goes **negative** past its first zero. A density can't do that. This repo treats positivity as a hard validity gate, not a footnote:

- `truncate=True` (default) kills the negative lobe by construction.
- `positivity_audit()` scans ρ(r) for any parameter set — best fits *and* posterior draws.
- `examples/positivity_audit.py` demonstrates the failure modes.

A fit that goes negative anywhere is unphysical, no matter how good its χ².

## Quick start

```bash
pip install -r requirements.txt
python examples/synthetic_demo.py    # generate + fit a synthetic curve
python examples/positivity_audit.py  # the negative-lobe check
python -m pytest tests/              # 13 tests
```

## What's here

- `tapered_sinc/profiles.py` — tapered-sinc, NFW, Burkert + positivity auditing
- `tapered_sinc/rotation_curve.py` — DM + baryons → model circular velocity
- `tapered_sinc/fitting.py` — LSQ init + emcee MCMC (bring your SPARC data)
- `tapered_sinc/comparison.py` — χ²/AIC/BIC on a consistent footing
- `tapered_sinc/validation.py` — residuals, held-out χ², parameter-grid audits
- `docs/MODEL.md` — the math, assumptions, and caveats
- `docs/PILOT_RESULTS.md` — honest 4-galaxy summary: 2 wins, 1 narrow, 1 loss

## Status

Pilot stage. Four SPARC galaxies analyzed (DDO154, IC2574, NGC2403, UGC128); expansion to 10–20 with held-out validation is the path to publication. See `docs/PILOT_RESULTS.md` for the unvarnished scorecard.

## Citation

```bibtex
@software{vasquez2026taperedsinc,
  author = {Vasquez, Brandon Manuel},
  title = {tapered-sinc-dark-matter: A Unified Tapered-Sinc Halo Model},
  year = {2026},
  url = {https://github.com/vasquezbrandon205/tapered-sinc-dark-matter}
}
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
