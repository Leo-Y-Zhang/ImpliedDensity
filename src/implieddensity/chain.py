"""Fetch and parse a live option chain from Cboe's public delayed quotes.

Cboe publish a full delayed chain as JSON with no key and no scraping. Yahoo's
endpoint now returns 401 without a session crumb, so it is not used.

Quotes are cached to disk on first fetch so that analysis and tests are
reproducible and the endpoint is not hit repeatedly.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import urllib.request

import numpy as np

URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/{symbol}.json"
CACHE_DIR = os.environ.get(
    "IMPLIEDDENSITY_CACHE",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "cache"))

# e.g. SPY260831C00420000 -> root SPY, 2026-08-31, call, strike 420.0
OCC = re.compile(r"^(?P<root>[A-Z]+)(?P<y>\d{2})(?P<m>\d{2})(?P<d>\d{2})"
                 r"(?P<cp>[CP])(?P<strike>\d{8})$")


def parse_occ(symbol):
    """Split an OCC option symbol. Returns None if it does not parse."""
    m = OCC.match(symbol)
    if not m:
        return None
    g = m.groupdict()
    return {
        "root": g["root"],
        "expiry": dt.date(2000 + int(g["y"]), int(g["m"]), int(g["d"])),
        "right": g["cp"],
        "strike": int(g["strike"]) / 1000.0,
    }


def fetch(symbol="SPY", use_cache=True):
    """Return the raw Cboe payload for a symbol, caching it on disk."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"cboe_{symbol}.json")
    if use_cache and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    req = urllib.request.Request(
        URL.format(symbol=symbol),
        headers={"User-Agent": "implieddensity/0.1 (research)"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    return payload


def expiries(payload):
    """Sorted list of expiries present in a payload."""
    out = set()
    for row in payload["data"]["options"]:
        p = parse_occ(row["option"])
        if p:
            out.add(p["expiry"])
    return sorted(out)


def calls_for_expiry(payload, expiry, min_bid=0.05, require_two_sided=True):
    """Extract the call side for one expiry as arrays.

    Rows with no bid are dropped: a zero bid means nobody will buy it, so the
    mid is not a price anyone would trade at, and deep out-of-the-money zero-bid
    quotes are exactly what makes a naive second derivative explode.
    """
    strikes, mids, bids, asks, spreads = [], [], [], [], []
    for row in payload["data"]["options"]:
        p = parse_occ(row["option"])
        if not p or p["right"] != "C" or p["expiry"] != expiry:
            continue
        bid, ask = float(row["bid"]), float(row["ask"])
        if require_two_sided and (bid <= 0 or ask <= 0):
            continue
        if bid < min_bid or ask < bid:
            continue
        strikes.append(p["strike"])
        mids.append(0.5 * (bid + ask))
        bids.append(bid)
        asks.append(ask)
        spreads.append(ask - bid)

    order = np.argsort(strikes)
    return {
        "strike": np.asarray(strikes)[order],
        "mid": np.asarray(mids)[order],
        "bid": np.asarray(bids)[order],
        "ask": np.asarray(asks)[order],
        "spread": np.asarray(spreads)[order],
    }


def spot_price(payload):
    return float(payload["data"]["close"])


def year_fraction(expiry, asof=None, basis=365.0):
    asof = asof or dt.date.today()
    return max((expiry - asof).days, 0) / basis
