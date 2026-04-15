#!/usr/bin/env python3
"""Compatibility wrapper for the unified baseline driver."""

import os
import sys


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from medmatch.experiments.baseline import run_baseline


def main():
    run_baseline(
        backend_name="local",
        selected_sheets=["IV push (17)", "IV continuous (16)"],
        max_entries_per_sheet=int(os.environ.get("MEDMATCH_MAX_ENTRIES", "5")),
        start_dir=ROOT,
    )


if __name__ == "__main__":
    main()
