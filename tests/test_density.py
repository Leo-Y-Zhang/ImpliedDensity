# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Tests for the density estimator.

The central one is a round trip: price a chain with Black-Scholes at a constant
volatility, run the recovered-density machinery over those prices, and check it
returns the analytic lognormal. If the estimator cannot recover a density it
was literally given, nothing it says about a real chain is worth reading.
"""
from __future__ import annotations

import datetime as dt
import json
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
    lognormal_cdf,
    lognormal_pdf,
    put_price,
    vega,
)
from implieddensity.chain import parse_occ, year_fraction  # noqa: E402
from implieddensity.density import (  # noqa: E402
    left_tail_probability,
    moments,
    risk_neutral_density,
    risk_neutral_density_svi,
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

    def test_left_tail_integrates_up_to_the_threshold_between_grid_points(self):
        """A threshold between two grid points must be integrated to exactly,
        not to the last grid point below it. On a uniform density over [0, 1]
        with ten intervals, P(X <= 0.15) is 0.15, not 0.1."""
        x = np.linspace(0.0, 1.0, 11)
        pdf = np.ones_like(x)
        self.assertAlmostEqual(left_tail_probability(x, pdf, 0.15), 0.15, places=12)
        self.assertAlmostEqual(left_tail_probability(x, pdf, 0.0), 0.0, places=12)
        self.assertAlmostEqual(left_tail_probability(x, pdf, 1.0), 1.0, places=12)

    def test_left_tail_includes_the_mass_below_the_grid(self):
        """The density is normalised on the strike range, so it describes the
        distribution conditional on finishing inside it. The unconditional tail
        probability is the mass below the range plus the in-range mass times
        the conditional one: 0.02 + (1 - 0.02 - 0.03) * 0.5 here."""
        x = np.linspace(0.0, 1.0, 11)
        pdf = np.ones_like(x)
        p = left_tail_probability(x, pdf, 0.5, mass_below=0.02, mass_above=0.03)
        self.assertAlmostEqual(p, 0.02 + 0.95 * 0.5, places=12)

    def test_left_tail_below_the_grid_is_refused_when_mass_lies_there(self):
        x = np.linspace(0.0, 1.0, 11)
        pdf = np.ones_like(x)
        self.assertEqual(left_tail_probability(x, pdf, -0.5), 0.0)
        with self.assertRaises(ValueError):
            left_tail_probability(x, pdf, -0.5, mass_below=0.02)


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


class TestDiscountingAndDividends(unittest.TestCase):
    """The published run uses r = 4.1% and a 1.2% dividend yield, but every
    round trip above has no dividend, and normalising the density cancels the
    e^{rT} factor. These pin both down on a chain where the answer is known."""

    s, r, t, v = 100.0, 0.05, 1.0, 0.22

    def test_black_scholes_with_a_dividend_yield_matches_a_textbook_value(self):
        # Hull, Options, Futures and Other Derivatives, index option example:
        # S=930, K=900, r=8%, q=3%, vol=20%, T=2 months gives c = 51.83
        c = float(call_price(930.0, 900.0, 0.08, 2.0 / 12.0, 0.20, 0.03))
        self.assertAlmostEqual(c, 51.83, delta=0.01)

    def test_zero_vol_call_is_the_discounted_forward_intrinsic(self):
        c = float(call_price(100.0, 90.0, 0.05, 1.0, 0.0, 0.03))
        self.assertAlmostEqual(c, 100.0 * math.exp(-0.03) - 90.0 * math.exp(-0.05),
                               places=12)

    def test_lognormal_mean_is_the_forward(self):
        x = np.linspace(1.0, 400.0, 20001)
        pdf = lognormal_pdf(x, self.s, self.r, self.t, self.v, div_yield=0.04)
        self.assertAlmostEqual(float(np.trapezoid(x * pdf, x)),
                               self.s * math.exp((self.r - 0.04) * self.t), places=4)

    def test_raw_integral_is_the_probability_mass_on_the_strike_range(self):
        """Before normalisation e^{rT} C''(K) integrates to the risk-neutral
        probability of finishing inside the strike range. Dropping or inverting
        the discount factor moves it by e^{±rT}, about 5% here."""
        strikes = np.arange(30.0, 260.0, 1.0)
        prices = call_price(self.s, strikes, self.r, self.t, self.v)
        x = np.linspace(strikes[0], strikes[-1], 20001)
        mass = float(np.trapezoid(lognormal_pdf(x, self.s, self.r, self.t, self.v), x))
        for fit in (risk_neutral_density, risk_neutral_density_svi):
            with self.subTest(method=fit.__name__):
                _, _, diag = fit(strikes, prices, self.s, self.r, self.t)
                self.assertAlmostEqual(diag["raw_integral"], mass, delta=2e-3)

    def test_mass_outside_the_strike_range_is_measured_from_the_call_slope(self):
        """P(S_T < K) = 1 + e^{rT} dC/dK, so the mass below the lowest strike
        and above the highest is known exactly even though the density there is
        not. On a lognormal chain both must match the analytic tails."""
        strikes = np.arange(70.0, 140.0, 1.0)
        prices = call_price(self.s, strikes, self.r, self.t, self.v, 0.02)
        below = float(lognormal_cdf(strikes[0], self.s, self.r, self.t, self.v, 0.02))
        above = 1.0 - float(lognormal_cdf(strikes[-1], self.s, self.r, self.t, self.v, 0.02))
        self.assertGreater(below, 0.04)   # the truncation is not negligible here
        for fit in (risk_neutral_density, risk_neutral_density_svi):
            with self.subTest(method=fit.__name__):
                _, _, diag = fit(strikes, prices, self.s, self.r, self.t, 0.02)
                self.assertAlmostEqual(diag["mass_below_grid"], below, delta=1e-4)
                self.assertAlmostEqual(diag["mass_above_grid"], above, delta=1e-4)

    def test_tail_probability_on_a_truncated_chain_matches_the_lognormal(self):
        """End to end: a lognormal chain quoted only from 70 up must still give
        the analytic P(S_T < 80) once the mass below 70 is added back. Before
        it was, the normalised density understated it substantially."""
        strikes = np.arange(70.0, 140.0, 1.0)
        prices = call_price(self.s, strikes, self.r, self.t, self.v)
        truth = float(lognormal_cdf(80.0, self.s, self.r, self.t, self.v))
        for fit in (risk_neutral_density, risk_neutral_density_svi):
            with self.subTest(method=fit.__name__):
                grid, q, diag = fit(strikes, prices, self.s, self.r, self.t)
                p = left_tail_probability(grid, q, 80.0, diag["mass_below_grid"],
                                          diag["mass_above_grid"])
                self.assertAlmostEqual(p, truth, delta=5e-4)

    def test_lognormal_cdf_is_the_integral_of_the_pdf(self):
        x = np.linspace(1e-6, 90.0, 200001)
        pdf = lognormal_pdf(x, self.s, self.r, self.t, self.v, 0.03)
        self.assertAlmostEqual(
            float(lognormal_cdf(90.0, self.s, self.r, self.t, self.v, 0.03)),
            float(np.trapezoid(pdf, x)), places=7)


    def test_round_trip_with_a_dividend_yield(self):
        """The dividend moves the forward below spot here (q > r), so a sign
        error in the forward, d1 or the lognormal shows up as a shifted mean."""
        q = 0.07
        strikes = np.arange(40.0, 220.0, 1.0)
        prices = call_price(self.s, strikes, self.r, self.t, self.v, q)
        forward = self.s * math.exp((self.r - q) * self.t)
        for fit in (risk_neutral_density, risk_neutral_density_svi):
            with self.subTest(method=fit.__name__):
                grid, dens, diag = fit(strikes, prices, self.s, self.r, self.t, q)
                self.assertAlmostEqual(diag["forward"], forward, places=9)
                self.assertAlmostEqual(diag["mean_vs_forward"], 0.0, delta=2e-3)
                truth = lognormal_pdf(grid, self.s, self.r, self.t, self.v, q)
                truth = truth / np.trapezoid(truth, grid)
                core = truth > truth.max() * 0.01
                rel = np.abs(dens[core] - truth[core]) / truth[core].max()
                self.assertLess(float(rel.max()), 0.05)


class TestPublishedResults(unittest.TestCase):
    """results.json records the fitted SVI smile, so every published tail
    probability can be recomputed offline from it. This pins the published
    numbers to the code: change either and this fails."""

    def test_published_tail_probabilities_follow_from_the_committed_smile(self):
        from implieddensity.analysis import tail_probabilities_from_results
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results.json")
        with open(path, encoding="utf-8") as fh:
            res = json.load(fh)
        recomputed = tail_probabilities_from_results(res)
        self.assertEqual(len(recomputed), 4)
        for key, value in recomputed.items():
            with self.subTest(key=key):
                self.assertAlmostEqual(res[key], value, places=9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
