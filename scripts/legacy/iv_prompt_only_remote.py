#!/usr/bin/env python3
"""Compatibility wrapper for the unified baseline driver."""

import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from medmatch.experiments.baseline import run_baseline


def main():
    run_baseline(
        backend_name="remote",
        selected_sheets=["IV intermittent (16)", "IV push (17)", "IV continuous (16)"],
        max_entries_per_sheet=int(os.environ.get("MEDMATCH_MAX_ENTRIES", "0")),
        start_dir=ROOT,
    )


if __name__ == "__main__":
    main()
