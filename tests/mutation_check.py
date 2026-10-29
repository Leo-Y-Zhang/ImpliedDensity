# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Selected, reproducible first-order mutations, each run in an isolated copy.

Run with the same Python environment used for CI (including ruff):
    python tests/mutation_check.py before
    python tests/mutation_check.py after

``before`` runs the CI gate with only the original test files; ``after`` runs it
with every test file. The original sources are never edited, and a timeout is
reported separately rather than counted as a kill. This is a targeted
experiment, not an exhaustive mutation score.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_TESTS = {"test_density.py"}
C, BS, D, S = ("src/implieddensity/chain.py", "src/implieddensity/blackscholes.py",
               "src/implieddensity/density.py", "src/implieddensity/svi.py")
CALL_FILTER = 'if not p or p["right"] != "C" or p["expiry"] != expiry:'
QUOTE_FILTER = "if bid < min_bid or ask < bid:"
BAND = "if not np.isfinite(price) or price < lower - 1e-10 or price > upper + 1e-10:"

# (module, identifier, original, replacement). Each original must occur once.
MUTANTS = [
    (C, "puts-included", CALL_FILTER, 'if not p or p["expiry"] != expiry:'),
    (C, "other-expiries-included", CALL_FILTER, 'if not p or p["right"] != "C":'),
    (C, "one-sided-accepted", "(bid <= 0 or ask <= 0)", "(bid <= 0 and ask <= 0)"),
    (C, "crossed-accepted", QUOTE_FILTER, "if bid < min_bid:"),
    (C, "min-bid-exclusive", QUOTE_FILTER, "if bid <= min_bid or ask < bid:"),
    (C, "mid-is-ask", "mids.append(0.5 * (bid + ask))", "mids.append(ask)"),
    (C, "spread-negated", "spreads.append(ask - bid)", "spreads.append(bid - ask)"),
    (C, "strikes-unsorted", "order = np.argsort(strikes)", "order = np.arange(len(strikes))"),
    (C, "strike-scale", 'int(g["strike"]) / 1000.0', 'int(g["strike"]) / 100.0'),
    (C, "refresh-ignored", "if use_cache and os.path.exists(path):", "if os.path.exists(path):"),
    (C, "year-fraction-negative", "max((expiry - asof).days, 0) / basis", "(expiry - asof).days / basis"),
    (C, "expiries-descending", "return sorted(out)", "return sorted(out, reverse=True)"),
    (BS, "zero-vol-not-degenerate", "degenerate = (tau <= 0) | (vol <= 0)", "degenerate = (tau <= 0)"),
    (BS, "no-upper-band", BAND, "if not np.isfinite(price) or price < lower - 1e-10:"),
    (BS, "no-lower-band", BAND, "if not np.isfinite(price) or price > upper + 1e-10:"),
    (BS, "loose-tolerance", "if b - a < tol:", "if b - a < 1e-3:"),
    (BS, "vega-no-sqrt-tau", "_norm_pdf(d1) * np.sqrt(tau)", "_norm_pdf(d1)"),
    (BS, "cdf-wrong-tail", "_norm_cdf(-d2), 0.0)", "_norm_cdf(d2), 0.0)"),
    (BS, "d1-ignores-dividend", "(rate - div_yield + 0.5 * vol ** 2)", "(rate + 0.5 * vol ** 2)"),
    (D, "density-discount-inverted", "disc = np.exp(rate * tau)", "disc = np.exp(-rate * tau)"),
    (D, "svi-forward-dividend-sign", "forward = spot * np.exp((rate - div_yield) * tau)",
     "forward = spot * np.exp((rate + div_yield) * tau)"),
    (D, "arbitrage-flag-always-true", '"butterfly_arbitrage_free": bool(np.min(g) >= 0)',
     '"butterfly_arbitrage_free": True'),
    (S, "svi-wrong-root-sign", "rho * (k - m) + np.sqrt", "rho * (k - m) - np.sqrt"),
    (S, "svi-five-points", "if len(k) < 6:", "if len(k) < 5:"),
    (S, "butterfly-missing-half", "term1 - term2 + d2w / 2.0", "term1 - term2 + d2w"),
]


def gate_commands():
    python = sys.executable
    return [
        [python, "scripts/check_spdx.py"],
        [python, "-m", "ruff", "check", "."],
        [python, "-m", "unittest", "discover", "-f", "-s", "tests"],
    ]


def run_gate(workspace):
    env = {**os.environ, "PYTHONPATH": str(workspace / "src"),
           "PYTHONDONTWRITEBYTECODE": "1", "IMPLIEDDENSITY_CACHE": str(workspace / "cache")}
    for command in gate_commands():
        label = "python " + " ".join(command[1:])
        try:
            result = subprocess.run(command, cwd=workspace, env=env, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    timeout=300)
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "command": label}
        if result.returncode:
            # Test names only. subTest parameters are dropped: they are noise
            # here, and a parameter like key='...' looks like a credential to a
            # secret scanner.
            failures = [line.split(" (")[0] for line in result.stdout.splitlines()
                        if line.startswith(("FAIL:", "ERROR:"))]
            return {"status": "killed", "command": label, "failures": failures}
    return {"status": "survived"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("before", "after"))
    parser.add_argument("--only", nargs="+", help="rerun named mutants and merge their results")
    args = parser.parse_args()
    if args.only and set(args.only) - {m[1] for m in MUTANTS}:
        parser.error("unknown mutant identifier")
    report_path = ROOT / "tests" / "mutation_results.json"
    report = (json.loads(report_path.read_text(encoding="utf-8"))
              if report_path.exists() else {})
    with tempfile.TemporaryDirectory(prefix=".mutation-", dir=ROOT) as temp:
        workspace = Path(temp)
        for directory in ("src", "tests", "scripts"):
            shutil.copytree(ROOT / directory, workspace / directory,
                            ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
        # Top-level files too: some tests read committed results beside the code.
        for path in ROOT.iterdir():
            if path.is_file():
                shutil.copy2(path, workspace)
        if args.phase == "before":
            for path in (workspace / "tests").glob("test_*.py"):
                if path.name not in ORIGINAL_TESTS:
                    path.unlink()
        baseline = run_gate(workspace)
        if baseline["status"] != "survived":
            raise SystemExit(f"Unmutated gate failed: {baseline}")
        results = []
        for module, identifier, original, replacement in MUTANTS:
            if args.only and identifier not in args.only:
                continue
            path = workspace / module
            source = path.read_text(encoding="utf-8")
            if source.count(original) != 1:
                raise SystemExit(f"{identifier}: expected exactly one mutation site")
            line = source[:source.index(original)].count("\n") + 1
            try:
                mutated = source.replace(original, replacement, 1)
                compile(mutated, module, "exec")
                path.write_text(mutated, encoding="utf-8")
                result = run_gate(workspace)
            finally:
                path.write_text(source, encoding="utf-8")
            results.append({"module": module, "id": identifier, "line": line, **result})
            print(f"{args.phase}: {module}:{line} {identifier}: {result['status']}", flush=True)
    merged = {r["id"]: r for r in report.get(args.phase, [])} if args.only else {}
    merged.update({r["id"]: r for r in results})
    report[args.phase] = [merged[m[1]] for m in MUTANTS if m[1] in merged]
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for module in dict.fromkeys(m[0] for m in MUTANTS):
        rows = [r for r in report[args.phase] if r["module"] == module]
        killed = sum(r["status"] == "killed" for r in rows)
        print(f"{module}: {killed}/{len(rows)} killed")


if __name__ == "__main__":
    main()
