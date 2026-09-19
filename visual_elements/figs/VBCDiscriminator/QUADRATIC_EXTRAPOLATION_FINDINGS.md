# Positive-field quadratic extrapolation

## Definition

For each cluster, `J2`, `D`, pinning source, and independently sorted NN
correlation, fit only the available positive-field points to

```text
C(h) = C0 + c1 h + c2 h^2 .
```

The measured `h=0` point is displayed for reference but is not used. The
rank-split source is excluded. The extrapolated splitting is

```text
Delta0 = C0_weakest - C0_strongest .
```

All current fits use `h=0.08,0.04,0.02,0.01`. A future `h=0.005` point is
included automatically.

## Current numerical result

With the default D6--D10 filter, there are currently 26 independent
`(cluster,J2,D,pin)` extrapolations and 78 individual correlation fits. The
worst individual `R^2` is `0.850017`; the largest RMSE is about `2.43e-3`.

At `J2=0.30`, Kuma and Izar agree well and should remain separate cross-checks:

| D | pin | Delta0 (Izar) | Delta0 (Kuma) |
|---:|---|---:|---:|
| 8 | dimer-plaquette | 0.247724 | 0.247831 |
| 8 | plaquette | 0.269329 | 0.271441 |
| 9 | dimer-plaquette | 0.242442 | 0.238134 |
| 9 | plaquette | 0.253113 | 0.254759 |

For `D=7`, only Kuma `J2=0.30` exists: `Delta0=0.258145` for the dimer pin and
`0.277341` for the plaquette pin.

The Izar trends are:

| J2 | D=8 dimer | D=8 plaq | D=9 dimer | D=9 plaq |
|---:|---:|---:|---:|---:|
| 0.29 | 0.267872 | 0.245641 | 0.228623 | 0.239451 |
| 0.30 | 0.247724 | 0.269329 | 0.242442 | 0.253113 |
| 0.31 | 0.257619 | 0.273313 | 0.260553 | 0.266351 |
| 0.32 | 0.306588 | 0.283280 | 0.267072 | 0.278426 |

The fit-only one-sigma errors on individual `Delta0` values are roughly
`0.00235--0.00751`, but they use only one residual degree of freedom and do not
include CTMRG, optimization, rank-switching, or fit-window systematics.

## Linear-response check

The coefficients must not be compared in isolation because they multiply
different powers of the field. The dimensionless diagnostic is evaluated at
the reference field `h_ref=0.02`:

```text
R(h_ref) = abs(c2*h_ref^2) / abs(c1*h_ref)
         = abs(c2/c1) * h_ref .
```

The quadratic correction is subleading at that scale when `R(0.02) < 1`.
All 26 current extrapolations pass this test for all three correlations; the
largest group-level ratio is `R(0.02)=0.2721`. At the largest field used in
the legacy fits, the maximum ratio is instead `1.0883`, so the high-field end
is not uniformly within the same linear-response regime.
The processed tables retain both this reference-field ratio and the analogous
ratio at the largest fitted field. Results are regenerated from the current
archive and should be read from
`data/processed/VBCPinningQuadraticExtrapolation`, rather than from a stale
hard-coded count in this note.

Dropping `h=0.08` and refitting the remaining three points shifts an
individual extrapolated `C0` by as much as `0.0138`. The conservative
strongest-plus-weakest bound on the corresponding splitting shift reaches
`0.0174`. This fit-window systematic is comparable to several differences
between the two pinned branches and is more important than the nominal fit
error.

## Current continuation campaign

The Izar campaign adds `h=0.005` after `0.08,0.04,0.02,0.01` for both
physical pins. D9 uses seven-day jobs at the selected seven J2 values. D8,
D7, and D6 use three-day jobs in the requested eleven-value priority order.
The automatic analysis excludes D5 and D11 by default.

The finite extrapolated splitting supports robust VBC bond order under both
pinning continuations. Splitting magnitude alone does **not** determine which
texture is the unbiased ground state; that still requires endpoint texture
and unbiased-energy information.
