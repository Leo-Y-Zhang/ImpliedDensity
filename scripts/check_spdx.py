# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""SPDX header check - one line per file, not a licence block.

The licence requires that LICENSE and NOTICE travel with the work. It does not
require the full legal text on top of every source file, and pasting it there
makes every file's first screenful identical and therefore unread. A single
machine-readable identifier does the same job and stays readable:

    # SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary

Run with --fix to insert missing headers. CI runs it without --fix, so a file
added without one fails the build rather than drifting in unnoticed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IDENTIFIER = "SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary"
SKIP_DIRS = {".venv", ".git", "__pycache__", ".pytest_cache", ".ruff_cache",
             "build", "dist", "cache"}


def source_files() -> list[Path]:
    out: list[Path] = []
    for pattern in ("src/**/*.py", "tests/**/*.py", "scripts/**/*.py"):
        for path in ROOT.glob(pattern):
            if not any(part in SKIP_DIRS for part in path.parts):
                out.append(path)
    return sorted(out)


def has_header(path: Path) -> bool:
    with path.open(encoding="utf-8") as fh:
        for _ in range(5):
            line = fh.readline()
            if not line:
                break
            if IDENTIFIER in line:
                return True
    return False


def insert(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    path.write_text(f"# {IDENTIFIER}\n" + text, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fix", action="store_true", help="insert missing headers")
    args = ap.parse_args()

    missing = [p for p in source_files() if not has_header(p)]
    if not missing:
        print(f"SPDX: all {len(source_files())} source files carry the identifier")
        return 0
    if args.fix:
        for p in missing:
            insert(p)
            print(f"  added header to {p.relative_to(ROOT)}")
        return 0
    print(f"SPDX: {len(missing)} file(s) missing the identifier:", file=sys.stderr)
    for p in missing:
        print(f"  {p.relative_to(ROOT)}", file=sys.stderr)
    print("\nrun: python scripts/check_spdx.py --fix", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
