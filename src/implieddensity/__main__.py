# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""ImpliedDensity CLI.

  implieddensity density [--symbol SPY] [--method svi|spline] [--refresh]
      Recover the risk-neutral density from a live option chain and print the
      moments, the tail probabilities, and the arbitrage diagnostics. Writes
      results.json, and density.png if matplotlib is installed.

  implieddensity chain [--symbol SPY] [--refresh]
      List the expiries available for a symbol with the number of two-sided
      call quotes at each, so you can see what there is before fitting it.

  implieddensity smile [--symbol SPY] [--target-days 120]
      Print the fitted SVI parameters and the implied-volatility smile at the
      quoted strikes, alongside the fit residual. Use this when a density looks
      wrong: a bad density is almost always a bad smile.

  implieddensity verify
      Run the offline checks that do not need a network: price a synthetic
      chain at a known constant volatility, recover the density, and require it
      to match the analytic lognormal. This is the one that decides whether any
      other output can be believed.

Quotes come from Cboe's public delayed-quote endpoint. Nothing here trades, and
no result should be relied on for a financial decision -- see the LICENSE.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys

import numpy as np


def _load(refresh: bool, symbol: str):
    from . import chain
    payload = chain.fetch(symbol, use_cache=not refresh)
    return chain, payload


def cmd_chain(args) -> int:
    chain, payload = _load(args.refresh, args.symbol)
    spot = chain.spot_price(payload)
    today = dt.date.today()
    print(f"{args.symbol}  spot {spot:,.2f}  as of {today}")
    print(f"{'expiry':>12} {'days':>6} {'two-sided calls':>16}")
    for e in chain.expiries(payload):
        days = (e - today).days
        if days < 0:
            continue
        n = len(chain.calls_for_expiry(payload, e)["strike"])
        if n:
            print(f"{e.isoformat():>12} {days:>6} {n:>16}")
    return 0


def cmd_smile(args) -> int:
    from .density import implied_vol_curve
    from .svi import butterfly_g, fit_svi, svi_implied_vol

    chain, payload = _load(args.refresh, args.symbol)
    spot = chain.spot_price(payload)
    expiry = _pick_expiry(chain, payload, args.target_days)
    tau = chain.year_fraction(expiry)
    calls = chain.calls_for_expiry(payload, expiry)

    strikes, vols, _ = implied_vol_curve(calls["strike"], calls["mid"], spot,
                                         args.rate, tau, args.div_yield)
    forward = spot * np.exp((args.rate - args.div_yield) * tau)
    params, _ = fit_svi(np.log(strikes / forward), vols ** 2 * tau)
    fitted = svi_implied_vol(strikes, forward, tau, params)
    g = butterfly_g(np.log(strikes / forward), params)

    print(f"{args.symbol}  expiry {expiry}  T {tau:.4f}y  forward {forward:,.2f}")
    print("SVI parameters: " + "  ".join(f"{k}={v:+.4f}" for k, v in params.items()))
    print(f"vol fit RMSE {float(np.sqrt(np.mean((fitted - vols) ** 2))):.5f}"
          f"   min g(k) {float(np.min(g)):+.4f}"
          f"   arbitrage-free {bool(np.min(g) >= 0)}")
    print(f"\n{'strike':>10} {'moneyness':>10} {'quoted vol':>11} {'SVI vol':>9} {'diff':>8}")
    step = max(1, len(strikes) // 25)
    for k, v, f in list(zip(strikes, vols, fitted))[::step]:
        print(f"{k:>10.1f} {k / forward:>10.3f} {v:>11.4f} {f:>9.4f} {f - v:>+8.4f}")
    return 0


def _pick_expiry(chain, payload, target_days: int):
    today = dt.date.today()
    options = [e for e in chain.expiries(payload) if (e - today).days >= 25]
    if not options:
        raise SystemExit("no expiry at least 25 days out in this chain")
    return min(options, key=lambda e: abs((e - today).days - target_days))


def cmd_density(args) -> int:
    from . import analysis
    analysis.run(symbol=args.symbol, rate=args.rate, div_yield=args.div_yield,
                 target_days=args.target_days, refresh=args.refresh,
                 method=args.method, quiet=False)
    return 0


def cmd_verify(args) -> int:
    """Offline correctness gate. No network, no cached data, no excuses."""
    import unittest
    loader = unittest.TestLoader()
    suite = loader.discover("tests")
    result = unittest.TextTestRunner(verbosity=2 if args.verbose else 1).run(suite)
    return 0 if result.wasSuccessful() else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="implieddensity",
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    sub = p.add_subparsers(dest="command")

    def common(sp):
        sp.add_argument("--symbol", default="SPY")
        sp.add_argument("--refresh", action="store_true",
                        help="re-fetch the chain instead of using the cache")
        sp.add_argument("--rate", type=float, default=0.041,
                        help="risk-free rate (default: a recent 3-month bill)")
        sp.add_argument("--div-yield", type=float, default=0.012)
        sp.add_argument("--target-days", type=int, default=120)
        return sp

    d = common(sub.add_parser("density", help="recover the risk-neutral density"))
    d.add_argument("--method", choices=("svi", "spline"), default="svi",
                   help="svi is arbitrage-checkable; spline interpolates quotes "
                        "exactly and is noisier")
    d.set_defaults(func=cmd_density)

    c = common(sub.add_parser("chain", help="list available expiries"))
    c.set_defaults(func=cmd_chain)

    s = common(sub.add_parser("smile", help="fitted SVI smile and residuals"))
    s.set_defaults(func=cmd_smile)

    v = sub.add_parser("verify", help="run the offline correctness suite")
    v.add_argument("--verbose", action="store_true")
    v.set_defaults(func=cmd_verify)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 2
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
