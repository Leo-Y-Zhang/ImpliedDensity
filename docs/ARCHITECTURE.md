# Architecture

The pipeline is a straight line: quotes in, implied vols, a fitted
smile, a call curve, its second derivative, a density out. Each stage
is a module, and the only stage with real freedom in it is the smile
fit -- which is where every difficulty lives.

## Module map

| module | responsibility |
|---|---|
| `chain.py` | fetch and cache the Cboe chain; parse OCC symbols |
| `blackscholes.py` | pricing, vega, implied vol by bisection |
| `svi.py` | five-parameter smile fit and the arbitrage condition |
| `density.py` | Breeden-Litzenberger, both smile fitters, moments |
| `analysis.py` | end-to-end run; writes `results.json` |
| `__main__.py` | CLI |


## Why it is shaped this way

**Implied vol by bisection, not Newton.** Vega collapses for deep in-
and out-of-the-money options, which is exactly where a quoted chain is
noisiest, and Newton diverges there. Bisection is slower and always
converges because the call price is monotone in volatility.

**Smoothing in vol space, not price space.** Implied vol is a smooth,
slowly varying function of strike; the call price is not. The density is
a second derivative, so any wiggle in the interpolant becomes a negative
region -- a butterfly arbitrage.

**A shape constraint rather than a filter.** SVI absorbs quote noise into
a five-parameter fit residual instead of into the curvature.

## What would break it

- A chain with fewer than eight invertible strikes raises rather than
  returning a shape nobody should trust.
- The grid is truncated at the traded strike range, so the mean falls
  slightly short of the forward. That shortfall is reported, not hidden.
  The density is normalised on that range, so it is the distribution
  conditional on finishing inside it; the moments are of that conditional
  distribution. Tail probabilities add back the mass outside the range,
  which the call slope gives exactly: P(S_T < K) = 1 + e^{rT} dC/dK.
- Cboe quotes are delayed, and the rate and dividend yield are assumed
  constants passed in by the caller.

