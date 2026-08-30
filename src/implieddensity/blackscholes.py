# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Black-Scholes pricing and implied volatility.

Kept deliberately small and dependency-light: the density estimator needs to go
from quoted prices to implied vol, smooth in vol space, and come back to prices,
so it needs both directions to be reliable rather than fast.
"""
from __future__ import annotations

import math

import numpy as np

SQRT_2PI = math.sqrt(2.0 * math.pi)


def _norm_cdf(x):
    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def _norm_pdf(x):
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * x * x) / SQRT_2PI


def d1_d2(spot, strike, rate, tau, vol, div_yield=0.0):
    spot, strike, vol, tau = map(lambda a: np.asarray(a, dtype=float),
                                 (spot, strike, vol, tau))
    with np.errstate(divide="ignore", invalid="ignore"):
        v = vol * np.sqrt(tau)
        d1 = (np.log(spot / strike) + (rate - div_yield + 0.5 * vol ** 2) * tau) / v
        d2 = d1 - v
    return d1, d2


def call_price(spot, strike, rate, tau, vol, div_yield=0.0):
    """European call under Black-Scholes. Handles tau=0 and vol=0 as limits."""
    spot, strike, vol, tau = map(lambda a: np.asarray(a, dtype=float),
                                 (spot, strike, vol, tau))
    intrinsic = np.maximum(spot * np.exp(-div_yield * tau)
                           - strike * np.exp(-rate * tau), 0.0)
    degenerate = (tau <= 0) | (vol <= 0)
    d1, d2 = d1_d2(spot, strike, rate, np.where(degenerate, 1.0, tau),
                   np.where(degenerate, 1.0, vol), div_yield)
    price = (spot * np.exp(-div_yield * tau) * _norm_cdf(d1)
             - strike * np.exp(-rate * tau) * _norm_cdf(d2))
    return np.where(degenerate, intrinsic, price)


def put_price(spot, strike, rate, tau, vol, div_yield=0.0):
    """From put-call parity, so parity holds exactly by construction."""
    c = call_price(spot, strike, rate, tau, vol, div_yield)
    return c - spot * np.exp(-div_yield * tau) + strike * np.exp(-rate * tau)


def vega(spot, strike, rate, tau, vol, div_yield=0.0):
    spot, strike, vol, tau = map(lambda a: np.asarray(a, dtype=float),
                                 (spot, strike, vol, tau))
    d1, _ = d1_d2(spot, strike, rate, tau, vol, div_yield)
    return spot * np.exp(-div_yield * tau) * _norm_pdf(d1) * np.sqrt(tau)


def implied_vol(price, spot, strike, rate, tau, div_yield=0.0,
                lo=1e-4, hi=5.0, tol=1e-8, max_iter=100):
    """Implied volatility by bisection.

    Bisection rather than Newton: vega collapses for deep in- and out-of-the-
    money options, which is exactly where a quoted chain is noisiest, and
    Newton diverges there. Bisection is slower and always converges because the
    call price is monotone in volatility.

    Returns NaN when the price is outside the no-arbitrage band, rather than
    silently returning a boundary value.
    """
    price = float(price)
    lower = float(call_price(spot, strike, rate, tau, 1e-9, div_yield))
    upper = float(call_price(spot, strike, rate, tau, hi, div_yield))
    if not np.isfinite(price) or price < lower - 1e-10 or price > upper + 1e-10:
        return float("nan")

    a, b = lo, hi
    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        if float(call_price(spot, strike, rate, tau, mid, div_yield)) < price:
            a = mid
        else:
            b = mid
        if b - a < tol:
            break
    return 0.5 * (a + b)


def lognormal_pdf(s, spot, rate, tau, vol, div_yield=0.0):
    """Analytic risk-neutral density under Black-Scholes.

    The benchmark the recovered density is compared against, and the ground
    truth the round-trip test uses.
    """
    s = np.asarray(s, dtype=float)
    m = math.log(spot) + (rate - div_yield - 0.5 * vol ** 2) * tau
    sd = vol * math.sqrt(tau)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.exp(-((np.log(s) - m) ** 2) / (2 * sd ** 2)) / (s * sd * SQRT_2PI)
    return np.where(s > 0, out, 0.0)
