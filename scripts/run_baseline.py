#!/usr/bin/env python3
"""Unified baseline CLI driver."""

import argparse
import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from medmatch.llm.config import SUPPORTED_BACKENDS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=SUPPORTED_BACKENDS, required=True)
    parser.add_argument("--category", default="all")
    parser.add_argument("--max-entries", type=int, default=int(os.environ.get("MEDMATCH_MAX_ENTRIES", "0")))
    parser.add_argument("--runs", type=int)
    args = parser.parse_args()

    category_map = {
        "all": None,
        "po_solid": ["PO Solid (40)"],
        "po_liquid": ["PO liquid (10)"],
        "iv_intermittent": ["IV intermittent (16)"],
        "iv_push": ["IV push (17)"],
        "iv_continuous": ["IV continuous (16)"],
    }
    selected_sheets = category_map.get(args.category)
    if args.category not in category_map:
        raise SystemExit(f"Unsupported category: {args.category}")

    from medmatch.experiments.baseline import run_baseline

    run_baseline(
        backend_name=args.backend,
        selected_sheets=selected_sheets,
        max_entries_per_sheet=args.max_entries,
        num_runs=args.runs,
        start_dir=ROOT,
    )


if __name__ == "__main__":
    main()
