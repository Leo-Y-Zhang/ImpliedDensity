"""SVI smile fitting (Gatheral's Stochastic Volatility Inspired parameterisation).

A cubic spline through quoted implied vols interpolates the smile but does not
constrain its curvature, and the density is a second derivative of the resulting
call curve. Small wiggles in the spline therefore become negative regions in the
density, which is a butterfly arbitrage: a portfolio with a guaranteed
non-negative payoff trading at a negative price.

SVI fits five parameters to the whole smile in total-variance space:

    w(k) = a + b * ( rho * (k - m) + sqrt((k - m)^2 + sigma^2) )

with k = log(K / F) the log-moneyness and w = implied variance * T. The form is
smooth by construction, has linear wings, and comes with a published condition
(Gatheral & Jacquier's g(k) >= 0) for the density it implies to be non-negative,
so the fit can be checked rather than hoped over.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


def svi_total_variance(k, a, b, rho, m, sigma):
    k = np.asarray(k, dtype=float)
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


def fit_svi(log_moneyness, total_variance, weights=None):
    """Least-squares fit of the five SVI parameters.

    Bounds keep the fit inside the region where the parameterisation is
    well posed: b >= 0, |rho| < 1, sigma > 0, and a + b*sigma*sqrt(1-rho^2) >= 0
    so total variance stays non-negative.
    """
    k = np.asarray(log_moneyness, dtype=float)
    w = np.asarray(total_variance, dtype=float)
    ok = np.isfinite(k) & np.isfinite(w) & (w > 0)
    k, w = k[ok], w[ok]
    if len(k) < 6:
        raise ValueError(f"need at least 6 usable points to fit SVI, got {len(k)}")
    wt = np.ones_like(w) if weights is None else np.asarray(weights)[ok]

    a0 = float(np.min(w))
    m0 = float(k[np.argmin(w)])
    span = max(float(k.max() - k.min()), 1e-3)
    x0 = [a0, 0.1, -0.5, m0, 0.1 * span]
    lo = [-np.inf, 0.0, -0.999, k.min() - span, 1e-4]
    hi = [np.inf, 10.0, 0.999, k.max() + span, 5.0 * span]

    def resid(p):
        return wt * (svi_total_variance(k, *p) - w)

    sol = least_squares(resid, x0, bounds=(lo, hi), max_nfev=20000)
    return dict(zip(("a", "b", "rho", "m", "sigma"), map(float, sol.x))), sol


def butterfly_g(k, p):
    """Gatheral & Jacquier's g(k). Negative anywhere means butterfly arbitrage.

    g(k) = (1 - k w'/(2w))^2 - (w'/4)(1/w + 1/4) w' + w''/2
    """
    k = np.asarray(k, dtype=float)
    a, b, rho, m, sigma = (p["a"], p["b"], p["rho"], p["m"], p["sigma"])
    root = np.sqrt((k - m) ** 2 + sigma ** 2)
    w = a + b * (rho * (k - m) + root)
    dw = b * (rho + (k - m) / root)
    d2w = b * sigma ** 2 / root ** 3
    with np.errstate(divide="ignore", invalid="ignore"):
        term1 = (1.0 - k * dw / (2.0 * w)) ** 2
        term2 = (dw ** 2 / 4.0) * (1.0 / w + 0.25)
        return term1 - term2 + d2w / 2.0


def svi_implied_vol(strikes, forward, tau, params):
    k = np.log(np.asarray(strikes, dtype=float) / forward)
    w = svi_total_variance(k, **params)
    return np.sqrt(np.maximum(w, 1e-12) / tau)
