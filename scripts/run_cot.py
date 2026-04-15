#!/usr/bin/env python3
"""Unified CoT CLI driver."""

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
    parser.add_argument("--category", default="iv")
    args = parser.parse_args()

    if args.category not in {"iv", "iv_push", "iv_intermittent"}:
        raise SystemExit("CoT currently supports IV-only categories during migration.")
    from medmatch.experiments.cot import run_cot

    selected = None
    if args.category == "iv_push":
        selected = ["IV push (17)"]
    elif args.category == "iv_intermittent":
        selected = ["IV intermittent (16)"]
    run_cot(
        backend_name=args.backend,
        selected_sheets=selected,
        max_entries_per_sheet=int(os.environ.get("MEDMATCH_MAX_ENTRIES", "0")),
        start_dir=ROOT,
    )


if __name__ == "__main__":
    main()
