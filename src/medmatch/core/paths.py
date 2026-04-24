"""Shared repository paths and version metadata for MedMatch datasets/results."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = REPO_ROOT / "data"
RESULTS_ROOT = REPO_ROOT / "results"

CURRENT_DATASET_VERSION = "medmatch2"
LEGACY_DATASET_VERSION = "med_match"
SCORER_VERSION = "v1"

CURRENT_DATA_DIR = DATA_ROOT / CURRENT_DATASET_VERSION
LEGACY_DATA_DIR = DATA_ROOT / LEGACY_DATASET_VERSION
CURRENT_RESULTS_ROOT = RESULTS_ROOT / CURRENT_DATASET_VERSION
CURRENT_WORKBOOK_PATH = REPO_ROOT / "datasets" / "MedMatch2.xlsx"
DATA_CHANGELOG_PATH = REPO_ROOT / "docs" / "DATA_CHANGELOG.md"


def default_data_dir() -> str:
    return str(CURRENT_DATA_DIR)


def current_results_root() -> str:
    return str(CURRENT_RESULTS_ROOT)


def dataset_version_for_path(path: str | Path) -> str:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = (REPO_ROOT / resolved).resolve()
    else:
        resolved = resolved.resolve()

    if resolved == CURRENT_DATA_DIR.resolve():
        return CURRENT_DATASET_VERSION
    if resolved == LEGACY_DATA_DIR.resolve():
        return LEGACY_DATASET_VERSION
    return resolved.name or "custom"
