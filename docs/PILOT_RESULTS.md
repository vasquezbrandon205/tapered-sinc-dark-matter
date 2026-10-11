# Pilot Results — Four SPARC Galaxies (honest summary)

From the November 2025 analysis draft (MCMC + LSQ, AIC/BIC).

| Galaxy | Type | sinc χ² | Best alternative | Verdict |
|---|---|---|---|---|
| DDO154 | dwarf | 16.3 | Burkert (10.2) | **Loss** — Burkert wins outright, fewer params |
| IC2574 | dwarf/LSB | 12.4 | Burkert (69.6) | **Clear win** — Δχ² ≈ 57, survives any penalty |
| NGC2403 | spiral | 429.4 | NFW (639.7) | **Win, with caveat** — χ²/dof ≈ 6 for everyone; errors underestimated, relative ranking only |
| UGC128 | giant LSB | 50.6 | NFW (60.6) | **Narrow win** — Δχ² = 10 with 3 extra params; AIC weakly favors, BIC ~ tie |

**Honest tally: 2 clear wins, 1 narrow win, 1 loss.** Not "competitive with
or superior across the sample" — the DDO154 row is a loss and should be
reported as one. A model comparison that hides its losses is less credible,
not more.

## Open items before publication

1. Positivity audit on all posterior draws (see `validation.py`).
2. Schive-soliton numerical comparison (anchor the physics claim).
3. Check `A ~ O(1)` in all posteriors.
4. Expand to 10–20 SPARC galaxies with a stated selection criterion.
5. Held-out validation: fit on a subset, evaluate on the rest.
6. Cite the SPARC model-comparison literature (Katz+ 2017, Li+ 2020).
