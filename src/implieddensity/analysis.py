# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""End-to-end run: chain in, risk-neutral density and diagnostics out.

Lives in the package rather than in a top-level script so that the CLI, the
tests and any future caller all go through one code path. Writes results.json,
which is the single source every number in the README is rendered from.
"""
from __future__ import annotations

import datetime as dt
import json

import numpy as np

from . import chain
from .blackscholes import implied_vol, lognormal_pdf
from .density import left_tail_probability, moments, risk_neutral_density, risk_neutral_density_svi


def pick_expiry(payload, target_days: int = 120):
    """Nearest expiry to the target that is at least 25 days out.

    The floor matters: inside a few weeks the smile is dominated by pinning and
    the strike grid coarsens, so a density fitted there says more about the
    expiry cycle than about the market's view.
    """
    today = dt.date.today()
    options = [e for e in chain.expiries(payload) if (e - today).days >= 25]
    if not options:
        raise SystemExit("no expiry at least 25 days out in this chain")
    return min(options, key=lambda e: abs((e - today).days - target_days))


def run(symbol="SPY", rate=0.041, div_yield=0.012, target_days=120,
        refresh=False, method="svi", quiet=False, out_path="results.json",
        figure_path="density.png"):
    payload = chain.fetch(symbol, use_cache=not refresh)
    spot = chain.spot_price(payload)
    expiry = pick_expiry(payload, target_days)
    tau = chain.year_fraction(expiry)
    calls = chain.calls_for_expiry(payload, expiry)

    say = (lambda *a: None) if quiet else print
    say(f"{symbol}  spot {spot:,.2f}  expiry {expiry}  T {tau:.4f}y  "
        f"two-sided call quotes {len(calls['strike'])}")

    fit = risk_neutral_density_svi if method == "svi" else risk_neutral_density
    grid, q, diag = fit(calls["strike"], calls["mid"], spot, rate, tau, div_yield)
    m = moments(grid, q)

    atm_i = int(np.argmin(np.abs(calls["strike"] - spot)))
    atm_vol = implied_vol(calls["mid"][atm_i], spot, calls["strike"][atm_i],
                          rate, tau, div_yield)
    ln = lognormal_pdf(grid, spot, rate, tau, atm_vol, div_yield)
    ln = ln / np.trapezoid(ln, grid)
    ln_m = moments(grid, ln)

    out = {
        "symbol": symbol, "method": method,
        "asof": dt.date.today().isoformat(),
        "spot": spot, "expiry": expiry.isoformat(),
        "days_to_expiry": (expiry - dt.date.today()).days,
        "tau_years": tau, "rate": rate, "div_yield": div_yield,
        "atm_implied_vol": float(atm_vol),
        "n_quotes": int(len(calls["strike"])),
        "median_spread": float(np.median(calls["spread"])),
        "diagnostics": diag, "implied": m, "lognormal": ln_m,
        "p_below_minus10pct": left_tail_probability(grid, q, spot * 0.90),
        "p_below_minus20pct": left_tail_probability(grid, q, spot * 0.80),
        "lognormal_p_below_minus10pct": left_tail_probability(grid, ln, spot * 0.90),
        "lognormal_p_below_minus20pct": left_tail_probability(grid, ln, spot * 0.80),
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)

    say(f"  strikes used {diag['n_strikes_used']} "
        f"({diag['strike_min']:.0f}-{diag['strike_max']:.0f})")
    say(f"  negative mass clipped {diag['negative_mass_before_clip']:.6f}")
    if "butterfly_g_min" in diag:
        say(f"  SVI vol RMSE          {diag['vol_rmse']:.5f}")
        say(f"  min g(k)              {diag['butterfly_g_min']:+.4f} "
            f"(arbitrage-free: {diag['butterfly_arbitrage_free']})")
    say(f"  mean vs forward       {diag['mean_vs_forward']:+.4%}")
    say(f"  implied   skew {m['skew']:+.3f}  excess kurtosis {m['excess_kurtosis']:+.3f}")
    say(f"  lognormal skew {ln_m['skew']:+.3f}  excess kurtosis {ln_m['excess_kurtosis']:+.3f}")
    say(f"  P(down 10%) implied {out['p_below_minus10pct']:.4f} "
        f"vs lognormal {out['lognormal_p_below_minus10pct']:.4f}")
    say(f"  P(down 20%) implied {out['p_below_minus20pct']:.4f} "
        f"vs lognormal {out['lognormal_p_below_minus20pct']:.4f}")

    try:
        plot(grid, q, ln, spot, out, figure_path)
        say(f"  wrote {figure_path}")
    except ImportError:
        say("  (figure skipped: matplotlib not installed; pip install '.[plot]')")
    return out


def plot(grid, q, ln, spot, out, path="density.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "figure.dpi": 160,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    fwd = out["diagnostics"]["forward"]

    ax.fill_between(grid, 0, q, where=grid <= spot * 0.9, color="#b3202c",
                    alpha=0.16, lw=0, label="10% or more down")
    ax.plot(grid, q, color="#b3202c", lw=1.9, label="implied by option prices")
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
    fig.savefig(path)
    plt.close(fig)
    return path
