# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Tests for the density estimator.

The central one is a round trip: price a chain with Black-Scholes at a constant
volatility, run the recovered-density machinery over those prices, and check it
returns the analytic lognormal. If the estimator cannot recover a density it
was literally given, nothing it says about a real chain is worth reading.
"""
from __future__ import annotations

import datetime as dt
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from implieddensity.blackscholes import (  # noqa: E402
    call_price,
    implied_vol,
    lognormal_pdf,
    put_price,
    vega,
)
from implieddensity.chain import parse_occ, year_fraction  # noqa: E402
from implieddensity.density import (  # noqa: E402
    left_tail_probability,
    moments,
    risk_neutral_density,
)


class TestBlackScholes(unittest.TestCase):
    def test_put_call_parity(self):
        s, k, r, t, v = 100.0, 95.0, 0.04, 0.5, 0.2
        c = float(call_price(s, k, r, t, v))
        p = float(put_price(s, k, r, t, v))
        self.assertAlmostEqual(c - p, s - k * math.exp(-r * t), places=9)

    def test_call_is_monotone_decreasing_in_strike(self):
        k = np.linspace(60, 140, 40)
        c = call_price(100.0, k, 0.03, 0.7, 0.25)
        self.assertTrue(np.all(np.diff(c) < 0))

    def test_call_is_monotone_increasing_in_vol(self):
        v = np.linspace(0.05, 1.5, 40)
        c = call_price(100.0, 100.0, 0.03, 0.7, v)
        self.assertTrue(np.all(np.diff(c) > 0))

    def test_call_within_no_arbitrage_bounds(self):
        s, k, r, t, v = 100.0, 110.0, 0.03, 0.8, 0.3
        c = float(call_price(s, k, r, t, v))
        self.assertGreaterEqual(c, max(s - k * math.exp(-r * t), 0.0) - 1e-9)
        self.assertLessEqual(c, s + 1e-9)

    def test_zero_time_is_intrinsic(self):
        self.assertAlmostEqual(float(call_price(120.0, 100.0, 0.05, 0.0, 0.3)),
                               20.0, places=9)
        self.assertAlmostEqual(float(call_price(80.0, 100.0, 0.05, 0.0, 0.3)),
                               0.0, places=9)

    def test_vega_is_positive_and_peaks_near_the_money(self):
        k = np.linspace(70, 130, 61)
        v = vega(100.0, k, 0.02, 0.5, 0.25)
        self.assertTrue(np.all(v >= 0))
        self.assertAlmostEqual(k[int(np.argmax(v))], 100.0, delta=6.0)


class TestImpliedVol(unittest.TestCase):
    def test_round_trip_recovers_the_input_vol(self):
        s, r, t = 100.0, 0.03, 0.6
        for k in (80.0, 95.0, 100.0, 110.0, 130.0):
            for v in (0.10, 0.25, 0.60):
                price = float(call_price(s, k, r, t, v))
                got = implied_vol(price, s, k, r, t)
                self.assertAlmostEqual(got, v, places=5,
                                       msg=f"K={k} vol={v}")

    def test_price_below_intrinsic_returns_nan(self):
        got = implied_vol(0.0, 120.0, 100.0, 0.05, 1.0)
        self.assertTrue(math.isnan(got))

    def test_price_above_spot_returns_nan(self):
        got = implied_vol(500.0, 100.0, 100.0, 0.03, 1.0)
        self.assertTrue(math.isnan(got))


class TestRoundTripDensity(unittest.TestCase):
    """Price a chain at constant vol, recover the density, compare to lognormal."""

    def setUp(self):
        self.s, self.r, self.t, self.v = 100.0, 0.03, 0.5, 0.22
        self.strikes = np.arange(55.0, 155.0, 1.0)
        self.prices = call_price(self.s, self.strikes, self.r, self.t, self.v)

    def test_recovers_the_lognormal_density(self):
        grid, q, diag = risk_neutral_density(
            self.strikes, self.prices, self.s, self.r, self.t)
        truth = lognormal_pdf(grid, self.s, self.r, self.t, self.v)
        truth = truth / np.trapezoid(truth, grid)
        # compare where the truth carries essentially all its mass
        core = truth > truth.max() * 0.01
        rel = np.abs(q[core] - truth[core]) / truth[core].max()
        self.assertLess(float(rel.max()), 0.05,
                        "recovered density departs from the known lognormal")

    def test_density_integrates_to_one(self):
        grid, q, _ = risk_neutral_density(
            self.strikes, self.prices, self.s, self.r, self.t)
        self.assertAlmostEqual(float(np.trapezoid(q, grid)), 1.0, places=6)

    def test_density_is_non_negative(self):
        _, q, _ = risk_neutral_density(
            self.strikes, self.prices, self.s, self.r, self.t)
        self.assertGreaterEqual(float(q.min()), 0.0)

    def test_mean_is_close_to_the_forward(self):
        """A risk-neutral density must price the forward correctly. The grid is
        truncated at the traded strike range, so a small shortfall is expected;
        a large one means the estimator is wrong."""
        grid, q, diag = risk_neutral_density(
            self.strikes, self.prices, self.s, self.r, self.t)
        self.assertAlmostEqual(diag["mean_vs_forward"], 0.0, delta=0.02)

    def test_recovered_lognormal_has_almost_no_skew(self):
        """Constant vol in, so the density should be a plain lognormal: mildly
        right-skewed in price space and nothing like an equity smile."""
        grid, q, _ = risk_neutral_density(
            self.strikes, self.prices, self.s, self.r, self.t)
        m = moments(grid, q)
        self.assertGreater(m["skew"], 0.0)
        self.assertLess(m["skew"], 0.8)

    def test_too_few_strikes_raises(self):
        with self.assertRaises(ValueError):
            risk_neutral_density(self.strikes[:4], self.prices[:4],
                                 self.s, self.r, self.t)


class TestSkewedSurfaceIsDetected(unittest.TestCase):
    """With a downward-sloping smile the density must come out left-skewed.

    This is the property the whole project claims about real equity chains, so
    it is tested against a surface where the answer is known by construction.
    """

    def test_negative_skew_shows_up_in_the_density(self):
        s, r, t = 100.0, 0.02, 0.5
        strikes = np.arange(55.0, 155.0, 1.0)
        # a falling smile: cheap upside, expensive downside protection
        vols = 0.30 - 0.0015 * (strikes - s)
        prices = call_price(s, strikes, r, t, vols)
        grid, q, _ = risk_neutral_density(strikes, prices, s, r, t)
        skewed = moments(grid, q)

        flat_prices = call_price(s, strikes, r, t, 0.30)
        gridf, qf, _ = risk_neutral_density(strikes, flat_prices, s, r, t)
        flat = moments(gridf, qf)

        self.assertLess(skewed["skew"], flat["skew"],
                        "a downward smile must skew the density left of flat vol")

    def test_far_left_tail_is_fatter_under_a_downward_smile(self):
        """The extra mass sits in the far tail, which is what a fat tail means.

        Measured here: at a spot of 100 the smile density carries more mass
        below 75 than the flat-vol one, and progressively more the further out
        you look (below 65 it is about half as much again). Nearer the money the
        comparison reverses -- below 80 the smile density is slightly thinner --
        because skew shifts the bulk right as well as stretching the tail. That
        is why a fat-tail claim has to be made at a genuinely extreme
        threshold, not a convenient one.
        """
        s, r, t = 100.0, 0.02, 0.5
        strikes = np.arange(55.0, 155.0, 1.0)
        vols = 0.30 - 0.0015 * (strikes - s)
        g1, q1, _ = risk_neutral_density(
            strikes, call_price(s, strikes, r, t, vols), s, r, t)
        g2, q2, _ = risk_neutral_density(
            strikes, call_price(s, strikes, r, t, 0.30), s, r, t)

        for threshold in (65.0, 70.0, 75.0):
            with self.subTest(threshold=threshold):
                self.assertGreater(left_tail_probability(g1, q1, threshold),
                                   left_tail_probability(g2, q2, threshold))

        # and the effect strengthens the further into the tail you go
        ratio_65 = (left_tail_probability(g1, q1, 65.0)
                    / left_tail_probability(g2, q2, 65.0))
        ratio_75 = (left_tail_probability(g1, q1, 75.0)
                    / left_tail_probability(g2, q2, 75.0))
        self.assertGreater(ratio_65, ratio_75)


class TestMoments(unittest.TestCase):
    def test_normal_density_moments(self):
        x = np.linspace(-12, 12, 4001)
        pdf = np.exp(-0.5 * x ** 2) / math.sqrt(2 * math.pi)
        m = moments(x, pdf)
        self.assertAlmostEqual(m["mean"], 0.0, places=6)
        self.assertAlmostEqual(m["sd"], 1.0, places=4)
        self.assertAlmostEqual(m["skew"], 0.0, places=5)
        self.assertAlmostEqual(m["excess_kurtosis"], 0.0, places=3)

    def test_left_tail_of_a_symmetric_density_is_half(self):
        x = np.linspace(-12, 12, 4001)
        pdf = np.exp(-0.5 * x ** 2) / math.sqrt(2 * math.pi)
        self.assertAlmostEqual(left_tail_probability(x, pdf, 0.0), 0.5, places=4)


class TestOccParsing(unittest.TestCase):
    def test_parses_a_real_symbol(self):
        got = parse_occ("SPY260831C00420000")
        self.assertEqual(got["root"], "SPY")
        self.assertEqual(got["expiry"], dt.date(2026, 8, 31))
        self.assertEqual(got["right"], "C")
        self.assertAlmostEqual(got["strike"], 420.0)

    def test_parses_a_fractional_strike(self):
        got = parse_occ("SPY261218P00437500")
        self.assertEqual(got["right"], "P")
        self.assertAlmostEqual(got["strike"], 437.5)

    def test_rejects_rubbish(self):
        self.assertIsNone(parse_occ("NOTASYMBOL"))
        self.assertIsNone(parse_occ(""))

    def test_year_fraction_is_never_negative(self):
        past = dt.date(2000, 1, 1)
        self.assertEqual(year_fraction(past, asof=dt.date(2026, 1, 1)), 0.0)

    def test_year_fraction_counts_days(self):
        t = year_fraction(dt.date(2026, 7, 1), asof=dt.date(2026, 1, 1))
        self.assertAlmostEqual(t, 181 / 365.0, places=9)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestSVI(unittest.TestCase):
    """SVI is the fix for the negative-density problem, so it gets its own tests."""

    def setUp(self):
        from implieddensity.svi import butterfly_g, fit_svi, svi_total_variance
        self.fit_svi = fit_svi
        self.w = svi_total_variance
        self.g = butterfly_g

    def test_recovers_its_own_parameters(self):
        truth = {"a": 0.02, "b": 0.15, "rho": -0.6, "m": 0.05, "sigma": 0.2}
        k = np.linspace(-0.6, 0.6, 60)
        w = self.w(k, **truth)
        got, _ = self.fit_svi(k, w)
        for name in truth:
            self.assertAlmostEqual(got[name], truth[name], places=3, msg=name)

    def test_fits_a_flat_smile_as_near_constant_variance(self):
        k = np.linspace(-0.5, 0.5, 50)
        w = np.full_like(k, 0.04)
        got, _ = self.fit_svi(k, w)
        fitted = self.w(k, **got)
        self.assertLess(float(np.max(np.abs(fitted - 0.04))), 1e-4)

    def test_too_few_points_raises(self):
        with self.assertRaises(ValueError):
            self.fit_svi(np.array([0.0, 0.1]), np.array([0.04, 0.041]))

    def test_butterfly_g_positive_for_a_benign_smile(self):
        p = {"a": 0.02, "b": 0.15, "rho": -0.4, "m": 0.0, "sigma": 0.3}
        self.assertGreater(float(np.min(self.g(np.linspace(-0.5, 0.5, 200), p))), 0)

    def test_svi_density_matches_the_lognormal_round_trip(self):
        from implieddensity.density import risk_neutral_density_svi
        s, r, t, v = 100.0, 0.03, 0.5, 0.22
        strikes = np.arange(55.0, 155.0, 1.0)
        prices = call_price(s, strikes, r, t, v)
        grid, q, diag = risk_neutral_density_svi(strikes, prices, s, r, t)
        truth = lognormal_pdf(grid, s, r, t, v)
        truth = truth / np.trapezoid(truth, grid)
        core = truth > truth.max() * 0.01
        rel = np.abs(q[core] - truth[core]) / truth[core].max()
        self.assertLess(float(rel.max()), 0.05)

    def test_svi_density_has_no_negative_mass_on_a_clean_chain(self):
        from implieddensity.density import risk_neutral_density_svi
        s, r, t = 100.0, 0.03, 0.5
        strikes = np.arange(55.0, 155.0, 1.0)
        vols = 0.30 - 0.0015 * (strikes - s)
        _, _, diag = risk_neutral_density_svi(
            strikes, call_price(s, strikes, r, t, vols), s, r, t)
        self.assertLess(diag["negative_mass_before_clip"], 1e-6)
        self.assertTrue(diag["butterfly_arbitrage_free"])
