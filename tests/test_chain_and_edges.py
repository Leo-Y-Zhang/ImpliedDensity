# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Edges the round-trip tests cannot see.

The round trips price a clean chain and recover it, so they never exercise what
the chain parser throws away, the limits of the pricing formulas, or a smile
that really does contain arbitrage. These tests do, each with an input built by
hand so the expected answer is known exactly.
"""
from __future__ import annotations

import datetime as dt
import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from implieddensity import chain  # noqa: E402
from implieddensity.blackscholes import call_price, vega  # noqa: E402
from implieddensity.density import risk_neutral_density_svi  # noqa: E402
from implieddensity.svi import (  # noqa: E402
    butterfly_g,
    fit_svi,
    svi_implied_vol,
    svi_total_variance,
)

EXPIRY = dt.date(2026, 12, 18)
LATER = dt.date(2027, 1, 15)


def occ(expiry, right, strike):
    return f"SPY{expiry:%y%m%d}{right}{int(round(strike * 1000)):08d}"


def payload(rows, close=600.0):
    return {"data": {"close": close, "options": [
        {"option": occ(e, r, k), "bid": b, "ask": a} for e, r, k, b, a in rows]}}


class TestCallsForExpiry(unittest.TestCase):
    def test_keeps_only_usable_calls_of_the_expiry_sorted_by_strike(self):
        p = payload([
            (EXPIRY, "C", 610.0, 4.0, 4.4),
            (EXPIRY, "C", 590.0, 15.0, 15.6),
            (EXPIRY, "P", 600.0, 8.0, 8.4),    # a put
            (LATER, "C", 600.0, 12.0, 12.5),   # another expiry
            (EXPIRY, "C", 640.0, 0.9, 0.8),    # crossed quote
            (EXPIRY, "C", 650.0, 0.04, 0.1),   # bid below the minimum
            (EXPIRY, "C", 660.0, 0.05, 0.1),   # bid exactly at the minimum
        ])
        got = chain.calls_for_expiry(p, EXPIRY)
        np.testing.assert_array_equal(got["strike"], [590.0, 610.0, 660.0])
        np.testing.assert_allclose(got["mid"], [15.3, 4.2, 0.075])
        np.testing.assert_allclose(got["bid"], [15.0, 4.0, 0.05])
        np.testing.assert_allclose(got["ask"], [15.6, 4.4, 0.1])
        np.testing.assert_allclose(got["spread"], [0.6, 0.4, 0.05])

    def test_one_sided_quotes_are_dropped_even_without_a_minimum_bid(self):
        p = payload([(EXPIRY, "C", 600.0, 0.0, 1.0), (EXPIRY, "C", 610.0, 0.2, 0.3)])
        got = chain.calls_for_expiry(p, EXPIRY, min_bid=0.0)
        np.testing.assert_array_equal(got["strike"], [610.0])
        both = chain.calls_for_expiry(p, EXPIRY, min_bid=0.0, require_two_sided=False)
        np.testing.assert_array_equal(both["strike"], [600.0, 610.0])

    def test_expiries_are_distinct_and_ascending_and_skip_unparseable_rows(self):
        p = payload([(LATER, "C", 600.0, 1.0, 2.0), (EXPIRY, "P", 600.0, 1.0, 2.0),
                     (EXPIRY, "C", 610.0, 1.0, 2.0)])
        p["data"]["options"].append({"option": "not an option", "bid": 1.0, "ask": 2.0})
        self.assertEqual(chain.expiries(p), [EXPIRY, LATER])
        self.assertEqual(chain.spot_price(p), 600.0)


class TestFetchCache(unittest.TestCase):
    def test_refresh_bypasses_the_cache_and_rewrites_it(self):
        fresh = {"data": {"close": 1.0, "options": []}}
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(chain, "CACHE_DIR", tmp):
            path = os.path.join(tmp, "cboe_SPY.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"stale": True}, fh)
            response = io.BytesIO(json.dumps(fresh).encode("utf-8"))
            with mock.patch.object(chain.urllib.request, "urlopen",
                                   return_value=response) as urlopen:
                self.assertEqual(chain.fetch("SPY", use_cache=True), {"stale": True})
                urlopen.assert_not_called()
                self.assertEqual(chain.fetch("SPY", use_cache=False), fresh)
                urlopen.assert_called_once()
            with open(path, encoding="utf-8") as fh:
                self.assertEqual(json.load(fh), fresh)


class TestBlackScholesLimits(unittest.TestCase):
    def test_zero_vol_exactly_at_the_forward_is_worth_nothing_not_nan(self):
        # log(S/K) = 0 and v = 0 make d1 = 0/0; the degenerate branch must win.
        self.assertEqual(float(call_price(100.0, 100.0, 0.0, 1.0, 0.0)), 0.0)

    def test_vega_is_the_derivative_of_the_price_in_vol(self):
        h = 1e-5
        for strike, tau in ((80.0, 0.25), (110.0, 0.5), (150.0, 2.0)):
            fd = (call_price(100.0, strike, 0.01, tau, 0.25 + h)
                  - call_price(100.0, strike, 0.01, tau, 0.25 - h)) / (2 * h)
            self.assertAlmostEqual(float(vega(100.0, strike, 0.01, tau, 0.25)),
                                   float(fd), places=5)


class TestSVIEdges(unittest.TestCase):
    P = {"a": 0.02, "b": 0.4, "rho": -0.3, "m": 0.05, "sigma": 0.2}

    def test_butterfly_g_matches_finite_differences(self):
        k = np.array([-0.4, -0.1, 0.0, 0.2, 0.5])
        h = 1e-4

        def w(x):
            return svi_total_variance(x, **self.P)

        dw = (w(k + h) - w(k - h)) / (2 * h)
        d2w = (w(k + h) - 2 * w(k) + w(k - h)) / h ** 2
        g = (1 - k * dw / (2 * w(k))) ** 2 - dw ** 2 / 4 * (1 / w(k) + 0.25) + d2w / 2
        np.testing.assert_allclose(butterfly_g(k, self.P), g, atol=1e-6)

    def test_fit_needs_six_points(self):
        k = np.linspace(-0.3, 0.3, 6)
        w = svi_total_variance(k, **self.P)
        fit_svi(k, w)
        with self.assertRaises(ValueError):
            fit_svi(k[:5], w[:5])

    def test_a_smile_with_butterfly_arbitrage_is_flagged(self):
        # The standard counterexample (Axel Vogt, quoted by Gatheral and Jacquier,
        # "Arbitrage-free SVI volatility surfaces"): g dips below zero.
        vogt = {"a": -0.0410, "b": 0.1331, "rho": 0.3060, "m": 0.3586, "sigma": 0.4153}
        strikes = np.linspace(40.0, 250.0, 60)
        prices = call_price(100.0, strikes, 0.0, 1.0,
                            svi_implied_vol(strikes, 100.0, 1.0, vogt))
        _, _, diag = risk_neutral_density_svi(strikes, prices, 100.0, 0.0, 1.0)
        self.assertLess(diag["butterfly_g_min"], 0.0)
        self.assertFalse(diag["butterfly_arbitrage_free"])


if __name__ == "__main__":
    unittest.main()
