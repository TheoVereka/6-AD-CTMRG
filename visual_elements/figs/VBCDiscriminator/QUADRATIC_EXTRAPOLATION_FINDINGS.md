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

There are 22 independent `(cluster,J2,D,pin)` extrapolations and 66
individual correlation fits. The worst individual `R^2` is `0.986544`; the
largest RMSE is about `2.3e-3`. Thus a quadratic describes the four sampled
points well in the residual sense.

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
`0.0024--0.0072`, but they use only one residual degree of freedom and do not
include CTMRG, optimization, rank-switching, or fit-window systematics.

## Linear-response check

Only 1 of 66 correlations satisfies the literal coefficient test
`abs(c2) < abs(c1)`, and no `(cluster,J2,D,pin)` combination passes it for all
three correlations. Because `c1` and `c2` multiply different powers of `h`, a
more direct measure on the sampled interval is
`abs(c2*h_max/c1)`. Its maximum is `0.557`: the quadratic contribution is not
dominant at `h=0.08`, but it is not negligible.

Dropping `h=0.08` and refitting the remaining three points shifts an
individual extrapolated `C0` by as much as `0.0138`. The conservative
strongest-plus-weakest bound on the corresponding splitting shift reaches
`0.0174`. This fit-window systematic is comparable to several differences
between the two pinned branches and is more important than the nominal fit
error.

## Minimal next calculations

1. No new `D=8` or `D=9` J2 grid is needed in `0.29--0.32`; it already exists.
2. To turn `D=7` into a J2 trend, first run both physical pins at `J2=0.29`
   and `0.32`. Run `J2=0.31` only if the two endpoints plus the existing
   `0.30` point are non-smooth.
3. If the literal `abs(c2)<abs(c1)` rule is mandatory, the present grid fails
   it and `h=0.005` is justified broadly. For a minimal diagnostic, add only
   the single `h=0.005` continuation stage for both pins at `D=8`,
   `J2=0.29` and `0.32`, where the ordering of the two extrapolated splittings
   reverses. This tests the most consequential feature with four short stages.
4. If that changes `Delta0` by more than the current fit-only error bars, add
   `h=0.005` for the remaining `D=8,9` points. Otherwise the existing D=9
   trend is already smooth enough for the present decision.

The finite extrapolated splitting supports robust VBC bond order under both
pinning continuations. Splitting magnitude alone does **not** determine which
texture is the unbiased ground state; that still requires endpoint texture
and unbiased-energy information.
