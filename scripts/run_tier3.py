#!/usr/bin/env python3
"""Unified Tier 3 CLI driver."""

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
    parser.add_argument("--category", choices=["oral", "po_solid", "po_liquid", "iv", "iv_push", "iv_intermittent"], required=True)
    args = parser.parse_args()
    from medmatch.experiments.tier3_normalize import run_tier3

    run_tier3(
        backend_name=args.backend,
        category=args.category,
        max_entries_per_sheet=int(os.environ.get("MEDMATCH_MAX_ENTRIES", "0")),
        start_dir=ROOT,
    )


if __name__ == "__main__":
    main()
