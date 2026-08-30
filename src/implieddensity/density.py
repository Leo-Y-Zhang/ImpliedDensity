"""Breeden-Litzenberger: recover the risk-neutral density from option prices.

The identity is q(K) = e^{rT} d2C/dK2. The whole difficulty is that a second
derivative amplifies noise, and quoted chains are noisy: wide spreads, stale
quotes, coarse strike grids. Differentiating raw mid prices produces a jagged
curve with negative regions, which is not a probability density.

The standard fix, and the one used here, is to smooth in implied-volatility
space rather than price space. Implied vol is a smooth, slowly varying function
of strike, so a spline through it is well behaved; converting back through
Black-Scholes gives a clean, arbitrage-respecting call curve to differentiate.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

from .blackscholes import call_price, implied_vol


def implied_vol_curve(strikes, prices, spot, rate, tau, div_yield=0.0):
    """Invert each quote to implied vol, dropping the ones that will not invert."""
    strikes = np.asarray(strikes, dtype=float)
    prices = np.asarray(prices, dtype=float)
    vols = np.array([implied_vol(p, spot, k, rate, tau, div_yield)
                     for k, p in zip(strikes, prices)])
    ok = np.isfinite(vols) & (vols > 1e-4) & (vols < 4.9)
    return strikes[ok], vols[ok], ok


def smooth_vol_spline(strikes, vols, smoothing_window=0):
    """Monotone-safe cubic spline through the vol smile in log-moneyness.

    Fitting in log-moneyness rather than raw strike keeps the knot spacing even
    across a chain whose strikes are denser near the money.
    """
    order = np.argsort(strikes)
    k = strikes[order]
    v = vols[order]
    # collapse duplicate strikes (calls and puts can both map to one strike)
    uniq, idx = np.unique(k, return_index=True)
    v = np.array([v[k == u].mean() for u in uniq])
    k = uniq
    if smoothing_window and smoothing_window > 1 and len(v) > smoothing_window:
        kernel = np.ones(smoothing_window) / smoothing_window
        v = np.convolve(v, kernel, mode="same")
        edge = smoothing_window // 2
        if edge:
            v[:edge] = v[edge]
            v[-edge:] = v[-edge - 1]
    return CubicSpline(np.log(k), v, extrapolate=True), k


def risk_neutral_density_svi(strikes, prices, spot, rate, tau, div_yield=0.0,
                             n_grid=800, k_lo=None, k_hi=None):
    """As risk_neutral_density, but with the smile fitted by SVI.

    The spline version interpolates the quoted vols exactly and inherits their
    noise as curvature, which shows up as negative density. SVI imposes a
    five-parameter shape on the whole smile, so noise is absorbed into the fit
    residual instead of the second derivative.
    """
    from .svi import fit_svi, svi_implied_vol, butterfly_g

    k_used, vols, _ = implied_vol_curve(strikes, prices, spot, rate, tau, div_yield)
    if len(k_used) < 8:
        raise ValueError(f"only {len(k_used)} strikes inverted to a usable "
                         "implied vol; need at least 8")

    forward = spot * np.exp((rate - div_yield) * tau)
    log_m = np.log(k_used / forward)
    params, sol = fit_svi(log_m, vols ** 2 * tau)

    lo = k_lo if k_lo is not None else k_used.min()
    hi = k_hi if k_hi is not None else k_used.max()
    grid = np.linspace(lo, hi, n_grid)

    vol_grid = np.clip(svi_implied_vol(grid, forward, tau, params), 1e-4, 5.0)
    c_grid = call_price(spot, grid, rate, tau, vol_grid, div_yield)
    q = np.exp(rate * tau) * np.gradient(np.gradient(c_grid, grid), grid)

    negative_mass = float(-np.trapezoid(np.minimum(q, 0.0), grid))
    q = np.maximum(q, 0.0)
    total = float(np.trapezoid(q, grid))
    if total <= 0:
        raise ValueError("recovered density has no positive mass")
    q = q / total

    g = butterfly_g(np.log(grid / forward), params)
    fitted = svi_implied_vol(k_used, forward, tau, params)
    diagnostics = {
        "method": "svi",
        "n_strikes_used": int(len(k_used)),
        "strike_min": float(k_used.min()),
        "strike_max": float(k_used.max()),
        "negative_mass_before_clip": negative_mass,
        "raw_integral": total,
        "mean": float(np.trapezoid(grid * q, grid)),
        "forward": float(forward),
        "svi_params": params,
        "vol_rmse": float(np.sqrt(np.mean((fitted - vols) ** 2))),
        "butterfly_g_min": float(np.min(g)),
        "butterfly_arbitrage_free": bool(np.min(g) >= 0),
    }
    diagnostics["mean_vs_forward"] = (diagnostics["mean"]
                                      / diagnostics["forward"] - 1.0)
    return grid, q, diagnostics


def risk_neutral_density(strikes, prices, spot, rate, tau, div_yield=0.0,
                         n_grid=800, k_lo=None, k_hi=None, smoothing_window=0):
    """Return (grid, density) with the density normalised to integrate to 1.

    Negative values produced by residual curvature noise are clipped to zero
    before normalising. Clipping is reported via ``diagnostics`` so that a
    heavily clipped result is visible rather than silently cleaned up.
    """
    k_used, vols, _ = implied_vol_curve(strikes, prices, spot, rate, tau, div_yield)
    if len(k_used) < 8:
        raise ValueError(f"only {len(k_used)} strikes inverted to a usable "
                         "implied vol; need at least 8")

    spline, k_sorted = smooth_vol_spline(k_used, vols, smoothing_window)
    lo = k_lo if k_lo is not None else k_sorted.min()
    hi = k_hi if k_hi is not None else k_sorted.max()
    grid = np.linspace(lo, hi, n_grid)

    vol_grid = spline(np.log(grid))
    vol_grid = np.clip(vol_grid, 1e-4, 5.0)
    c_grid = call_price(spot, grid, rate, tau, vol_grid, div_yield)

    d1 = np.gradient(c_grid, grid)
    d2 = np.gradient(d1, grid)
    q = np.exp(rate * tau) * d2

    negative_mass = float(-np.trapezoid(np.minimum(q, 0.0), grid))
    q = np.maximum(q, 0.0)
    total = float(np.trapezoid(q, grid))
    if total <= 0:
        raise ValueError("recovered density has no positive mass")
    q = q / total

    diagnostics = {
        "n_strikes_used": int(len(k_used)),
        "strike_min": float(k_sorted.min()),
        "strike_max": float(k_sorted.max()),
        "negative_mass_before_clip": negative_mass,
        "raw_integral": total,
        "mean": float(np.trapezoid(grid * q, grid)),
        "forward": float(spot * np.exp((rate - div_yield) * tau)),
    }
    diagnostics["mean_vs_forward"] = (diagnostics["mean"]
                                      / diagnostics["forward"] - 1.0)
    return grid, q, diagnostics


def moments(grid, density):
    """Mean, standard deviation, skewness and excess kurtosis of a density."""
    m1 = float(np.trapezoid(grid * density, grid))
    var = float(np.trapezoid((grid - m1) ** 2 * density, grid))
    sd = float(np.sqrt(var))
    if sd <= 0:
        return {"mean": m1, "sd": 0.0, "skew": float("nan"),
                "excess_kurtosis": float("nan")}
    skew = float(np.trapezoid(((grid - m1) / sd) ** 3 * density, grid))
    kurt = float(np.trapezoid(((grid - m1) / sd) ** 4 * density, grid)) - 3.0
    return {"mean": m1, "sd": sd, "skew": skew, "excess_kurtosis": kurt}


def left_tail_probability(grid, density, threshold):
    """Risk-neutral probability of finishing below ``threshold``."""
    mask = grid <= threshold
    if not mask.any():
        return 0.0
    return float(np.trapezoid(density[mask], grid[mask]))
