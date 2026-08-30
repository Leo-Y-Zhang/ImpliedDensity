"""Render README.md from README.template.md and results.json.

Fails loudly on a missing value so the README cannot quote a number the
analysis did not produce.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))


def count_tests():
    r = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests"],
                       cwd=HERE, capture_output=True, text=True, timeout=600)
    m = re.search(r"Ran (\d+) tests", r.stderr or r.stdout)
    if not m or r.returncode != 0:
        sys.exit("test suite must pass before the README is regenerated")
    return int(m.group(1))


def spline_comparison(res):
    """Re-derive the spline-vs-SVI comparison rather than quoting a past run."""
    sys.path.insert(0, os.path.join(HERE, "src"))
    import datetime as dt

    from implieddensity import chain
    from implieddensity.density import risk_neutral_density

    payload = chain.fetch(res["symbol"])
    spot = chain.spot_price(payload)
    expiry = dt.date.fromisoformat(res["expiry"])
    tau = res["tau_years"]
    calls = chain.calls_for_expiry(payload, expiry)
    _, _, diag = risk_neutral_density(calls["strike"], calls["mid"], spot,
                                      res["rate"], tau, res["div_yield"])
    return diag


res = json.load(open(os.path.join(HERE, "results.json"), encoding="utf-8"))
d = res["diagnostics"]
if d.get("method") != "svi":
    sys.exit("results.json was produced by the spline fitter; "
             "run `implieddensity density` (SVI is the default) first")
sp = spline_comparison(res)

V = {
    "symbol": res["symbol"],
    "asof": res["asof"],
    "spot": f"{res['spot']:,.2f}",
    "expiry": res["expiry"],
    "days": res["days_to_expiry"],
    "tau": f"{res['tau_years']:.3f}",
    "n_quotes": res["n_quotes"],
    "atm_vol": f"{res['atm_implied_vol']:.1%}",
    "skew": f"{res['implied']['skew']:+.2f}",
    "kurt": f"{res['implied']['excess_kurtosis']:+.2f}",
    "ln_skew": f"{res['lognormal']['skew']:+.2f}",
    "ln_kurt": f"{res['lognormal']['excess_kurtosis']:+.2f}",
    "p10": f"{res['p_below_minus10pct']:.2%}",
    "p20": f"{res['p_below_minus20pct']:.2%}",
    "ln_p10": f"{res['lognormal_p_below_minus10pct']:.2%}",
    "ln_p20": f"{res['lognormal_p_below_minus20pct']:.2%}",
    "p20_ratio": f"{res['p_below_minus20pct'] / res['lognormal_p_below_minus20pct']:.1f}",
    "svi_neg": f"{100 * d['negative_mass_before_clip'] / d['raw_integral']:.2f}",
    "svi_mean_err": f"{d['mean_vs_forward']:+.2%}",
    "gmin": f"{d['butterfly_g_min']:+.4f}",
    "rmse": f"{d['vol_rmse']:.5f}",
    "spline_neg": f"{100 * sp['negative_mass_before_clip'] / sp['raw_integral']:.1f}",
    "spline_mean_err": f"{sp['mean_vs_forward']:+.2%}",
    "n_tests": count_tests(),
}

tpl = open(os.path.join(HERE, "README.template.md"), encoding="utf-8").read()


def sub(m):
    k = m.group(1)
    if k not in V:
        raise KeyError(f"template needs '{k}' but it was not computed")
    return str(V[k])


out = re.sub(r"<<(\w+)>>", sub, tpl)
left = re.findall(r"<<[^>]*>>", out)
if left:
    sys.exit(f"unfilled placeholders: {left}")

open(os.path.join(HERE, "README.md"), "w", encoding="utf-8").write(out)
print(f"wrote README.md ({len(V)} values injected, {V['n_tests']} tests passing)")
