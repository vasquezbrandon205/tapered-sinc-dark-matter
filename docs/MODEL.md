# The Model

## Tapered-sinc density profile

```
rho(r) = w(r) * rho_core(r) + (1 - w(r)) * A * rho_outer(r)
```

- **Core**: `rho_core(r) = rho0 * sinc(pi * r / rc)`, with `sinc(x) = sin(x)/x`,
  regularized so `rho(0) = rho0` (finite central density — the cusp is gone).
- **Outer**: `rho_outer(r)` is NFW-like: `rhos / ((r/rs)(1 + r/rs)^2)`.
- **Taper**: `w(r) = 1 / (1 + exp((r - rc) / Delta))`, `Delta = eta * rc`
  with `eta ~ 0.1`. Smooth, monotonic, 1 deep inside the core, 0 far out.
- **A**: free amplitude matching inner/outer densities at the transition.
  Honest fits should find `A ~ O(1)`; large deviations mean the taper
  story is doing unacknowledged work.

## Why sinc (and when to distrust it)

Under steady-state, spherically symmetric assumptions with an effective
wave-like or dispersive pressure term in the core, the potential obeys a
Helmholtz-type equation `∇²Φ + k²Φ = 0`, whose regular spherical solution
is `Φ ∝ sin(kr)/r`. Poisson's equation then gives `rho ∝ sin(kr)/r` —
the sinc form. This is a **toy derivation**: it shows the sinc core is not
ad-hoc, but it is not a derivation from a specific microphysical model.

**Status: HYPOTHESIZED.** The honest anchor would be a numerical comparison
against the Schrödinger-Poisson soliton profile (Schive et al. 2014),
which is *not* exactly sinc. Until that comparison exists, treat the
microphysical story as plausible, not established.

## The negative-lobe problem

`sinc(x)` oscillates: it is negative for `pi < x < 2*pi`, i.e. the raw core
goes negative just past `r = rc`. A physical density cannot do that.

Two defenses, in order of strength:

1. **Hard truncation** (`truncate=True`, the default): the core term is
   exactly zero for `r > rc`. The negative lobe cannot exist by
   construction. The taper handles the handoff to the outer profile.
2. **Taper suppression** (`truncate=False`): relies on `w(r)` killing the
   core term before the lobe matters. Works for some parameters, fails
   for others (wide tapers, extreme amplitudes).

**Rule**: every fit must pass `positivity_audit()` — on the best fit AND
across posterior draws (`audit_parameter_grid`). A negative density
anywhere invalidates the fit regardless of chi-squared. See
`examples/positivity_audit.py`.

## Parameters

| Param | Meaning | Units | Prior (pilot) |
|---|---|---|---|
| `log10_rho0` | central density | log10(Msun/kpc³) | [3, 13] |
| `rc` | core radius | kpc | [0.01, 200] |
| `log10_rhos` | NFW scale density | log10(Msun/kpc³) | [3, 13] |
| `rs` | NFW scale radius | kpc | [0.1, 2000] |
| `log10_A` | outer amplitude | log10 | [-2, 2] |
| `log10_Ystar` | stellar M/L | log10 | [log10(0.01), log10(10)] |
| `eta` | taper width fraction | — | 0.1 (fixed) |

`eta` is fixed in the pilot; fitting it or testing sensitivity is open work.

## Rotation curves

```
V_model² = V_dm² + Ystar * V_disk² + Ystar * V_bulge² + V_gas²
```

Standard SPARC treatment: stellar disk/bulge scaled by mass-to-light ratio,
gas taken as given. `V_dm` from numerical enclosed mass `M(<r)`.

## Model comparison

- Same data, same error model, same Ystar treatment for every model.
- `chi2`, `AIC = 2k + chi2`, `BIC = k ln N + chi2`.
- Report ΔAIC/ΔBIC vs best, not raw values.
- If **all** models have `chi2/dof >> 1`, errors are underestimated
  (known SPARC issue) — trust relative rankings, not absolute chi2.
