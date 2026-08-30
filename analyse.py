"""Recover the market's risk-neutral density from a live SPY option chain.

Writes results.json and density.png. Every number quoted in the README comes
from results.json rather than being typed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
sys.stdout.reconfigure(encoding="utf-8")

from implieddensity import chain                     # noqa: E402
from implieddensity.blackscholes import lognormal_pdf, implied_vol  # noqa: E402
from implieddensity.density import (                 # noqa: E402
    risk_neutral_density, risk_neutral_density_svi, moments, left_tail_probability)


def pick_expiry(payload, target_days=120):
    today = dt.date.today()
    options = [e for e in chain.expiries(payload) if (e - today).days >= 25]
    if not options:
        raise SystemExit("no expiry far enough out in this chain")
    return min(options, key=lambda e: abs((e - today).days - target_days))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--rate", type=float, default=0.041,
                    help="risk-free rate; default is a recent 3-month bill")
    ap.add_argument("--div-yield", type=float, default=0.012)
    ap.add_argument("--target-days", type=int, default=120)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--method", choices=("svi", "spline"), default="svi",
                    help="smile fitter; svi is arbitrage-checkable, "
                         "spline interpolates quotes exactly and is noisier")
    args = ap.parse_args()

    payload = chain.fetch(args.symbol, use_cache=not args.refresh)
    spot = chain.spot_price(payload)
    expiry = pick_expiry(payload, args.target_days)
    tau = chain.year_fraction(expiry)
    calls = chain.calls_for_expiry(payload, expiry)

    print(f"{args.symbol}  spot {spot:.2f}  expiry {expiry}  "
          f"T {tau:.4f}y  two-sided call quotes {len(calls['strike'])}")

    fit = risk_neutral_density_svi if args.method == "svi" else risk_neutral_density
    grid, q, diag = fit(
        calls["strike"], calls["mid"], spot, args.rate, tau, args.div_yield)
    m = moments(grid, q)

    # a lognormal benchmark at the at-the-money implied vol
    atm_i = int(np.argmin(np.abs(calls["strike"] - spot)))
    atm_vol = implied_vol(calls["mid"][atm_i], spot, calls["strike"][atm_i],
                          args.rate, tau, args.div_yield)
    ln = lognormal_pdf(grid, spot, args.rate, tau, atm_vol, args.div_yield)
    ln = ln / np.trapezoid(ln, grid)
    ln_m = moments(grid, ln)

    down_10 = spot * 0.90
    down_20 = spot * 0.80
    out = {
        "symbol": args.symbol,
        "method": args.method,
        "asof": dt.date.today().isoformat(),
        "spot": spot,
        "expiry": expiry.isoformat(),
        "days_to_expiry": (expiry - dt.date.today()).days,
        "tau_years": tau,
        "rate": args.rate,
        "div_yield": args.div_yield,
        "atm_implied_vol": float(atm_vol),
        "n_quotes": int(len(calls["strike"])),
        "median_spread": float(np.median(calls["spread"])),
        "diagnostics": diag,
        "implied": m,
        "lognormal": ln_m,
        "p_below_minus10pct": left_tail_probability(grid, q, down_10),
        "p_below_minus20pct": left_tail_probability(grid, q, down_20),
        "lognormal_p_below_minus10pct": left_tail_probability(grid, ln, down_10),
        "lognormal_p_below_minus20pct": left_tail_probability(grid, ln, down_20),
    }
    with open("results.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)

    print(f"  strikes used {diag['n_strikes_used']} "
          f"({diag['strike_min']:.0f}-{diag['strike_max']:.0f})")
    print(f"  negative mass clipped {diag['negative_mass_before_clip']:.6f}")
    if "butterfly_g_min" in diag:
        print(f"  SVI vol RMSE          {diag['vol_rmse']:.5f}")
        print(f"  min g(k)              {diag['butterfly_g_min']:+.4f} "
              f"(arbitrage-free: {diag['butterfly_arbitrage_free']})")
    print(f"  mean vs forward       {diag['mean_vs_forward']:+.4%}")
    print(f"  implied   skew {m['skew']:+.3f}  excess kurtosis {m['excess_kurtosis']:+.3f}")
    print(f"  lognormal skew {ln_m['skew']:+.3f}  excess kurtosis {ln_m['excess_kurtosis']:+.3f}")
    print(f"  P(down 10%) implied {out['p_below_minus10pct']:.4f} "
          f"vs lognormal {out['lognormal_p_below_minus10pct']:.4f}")
    print(f"  P(down 20%) implied {out['p_below_minus20pct']:.4f} "
          f"vs lognormal {out['lognormal_p_below_minus20pct']:.4f}")

    try:
        plot(grid, q, ln, spot, out)
    except Exception as exc:  # pragma: no cover - plotting is optional
        print(f"  (figure skipped: {exc})")


def plot(grid, q, ln, spot, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "figure.dpi": 160,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    fwd = out["diagnostics"]["forward"]

    ax.fill_between(grid, 0, q, where=grid <= spot * 0.9, color="#b3202c",
                    alpha=0.16, lw=0, label="10% or more down")
    ax.plot(grid, q, color="#b3202c", lw=1.9,
            label="implied by option prices")
    ax.plot(grid, ln, color="#4a4f57", lw=1.4, ls=(0, (4, 3)),
            label=f"lognormal at ATM vol {out['atm_implied_vol']:.1%}")
    ax.axvline(fwd, color="#8a8f98", lw=0.9, ls=(0, (2, 3)))
    ax.text(fwd, ax.get_ylim()[1] * 0.96, " forward", fontsize=7.5,
            color="#5d646e", va="top")

    ax.set_xlim(grid.min(), min(grid.max(), spot * 1.5))
    ax.set_ylim(bottom=0)
    ax.set_xlabel(f"{out['symbol']} price at expiry ({out['expiry']})")
    ax.set_ylabel("risk-neutral density")
    ax.set_title(f"What the option market thinks {out['symbol']} will do\n"
                 f"skew {out['implied']['skew']:+.2f} against "
                 f"{out['lognormal']['skew']:+.2f} for a lognormal",
                 fontsize=9.5, loc="left")
    ax.legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig("density.png")
    print("  wrote density.png")


if __name__ == "__main__":
    main()
