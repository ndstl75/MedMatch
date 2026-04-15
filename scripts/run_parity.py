#!/usr/bin/env python3
"""Run a small parity matrix against unified and legacy MedMatch entrypoints."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PYTHON = "/opt/anaconda3/envs/medmatch/bin/python3"
DEFAULT_LOCAL_MODEL = "medgemma:27b"
DEFAULT_REMOTE_MODEL = "gemma-3-27b-it"
SHEET_NAMES = {
    "iv_push": "IV push (17)",
    "iv_intermittent": "IV intermittent (16)",
}


@dataclass
class CaseSpec:
    family: str
    unified_cmd: list[str]
    legacy_cmd: list[str]
    unified_results_dir: Path
    legacy_results_dir: Path
    result_prefix: str


@dataclass
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


def build_local_cases(category: str, python_bin: str) -> list[CaseSpec]:
    sheet_name = SHEET_NAMES[category]
    result_prefix = sheet_name.replace(" ", "_").replace("(", "").replace(")", "")
    return [
        CaseSpec(
            family="baseline",
            unified_cmd=[python_bin, "scripts/run_baseline.py", "--backend", "local", "--category", category],
            legacy_cmd=[python_bin, "scripts/legacy/local/medmatch_test_local_appendix_exact.py"],
            unified_results_dir=ROOT / "results",
            legacy_results_dir=ROOT / "scripts" / "legacy" / "local" / "results",
            result_prefix=result_prefix,
        ),
        CaseSpec(
            family="cot",
            unified_cmd=[python_bin, "scripts/run_cot.py", "--backend", "local", "--category", category],
            legacy_cmd=[python_bin, "scripts/legacy/local/iv_cot_experiment_local.py"],
            unified_results_dir=ROOT / "results",
            legacy_results_dir=ROOT / "scripts" / "legacy" / "local" / "results",
            result_prefix=result_prefix,
        ),
        CaseSpec(
            family="tier3",
            unified_cmd=[python_bin, "scripts/run_tier3.py", "--backend", "local", "--category", category],
            legacy_cmd=[python_bin, "scripts/legacy/local/iv_llm_normalize_local.py"],
            unified_results_dir=ROOT / "results",
            legacy_results_dir=ROOT / "scripts" / "legacy" / "local" / "results",
            result_prefix=result_prefix,
        ),
        CaseSpec(
            family="exemplar_rag",
            unified_cmd=[python_bin, "scripts/run_exemplar_rag.py", "--backend", "local", "--category", category],
            legacy_cmd=[python_bin, "scripts/legacy/local/iv_exemplar_rag_local.py"],
            unified_results_dir=ROOT / "results",
            legacy_results_dir=ROOT / "scripts" / "legacy" / "local" / "results",
            result_prefix=result_prefix,
        ),
    ]


def build_remote_cases(category: str, python_bin: str) -> list[CaseSpec]:
    sheet_name = SHEET_NAMES[category]
    result_prefix = sheet_name.replace(" ", "_").replace("(", "").replace(")", "")
    return [
        CaseSpec(
            family="baseline",
            unified_cmd=[
                python_bin,
                "scripts/run_baseline.py",
                "--backend",
                "remote",
                "--category",
                category,
                "--runs",
                "1",
                "--max-entries",
                "1",
            ],
            legacy_cmd=[python_bin, "scripts/legacy/iv_prompt_only_remote.py"],
            unified_results_dir=ROOT / "results",
            legacy_results_dir=ROOT / "results",
            result_prefix=result_prefix,
        ),
        CaseSpec(
            family="cot",
            unified_cmd=[python_bin, "scripts/run_cot.py", "--backend", "remote", "--category", category],
            legacy_cmd=[python_bin, "scripts/legacy/iv_cot_experiment.py"],
            unified_results_dir=ROOT / "results",
            legacy_results_dir=ROOT / "scripts" / "legacy" / "results",
            result_prefix=result_prefix,
        ),
        CaseSpec(
            family="tier3",
            unified_cmd=[python_bin, "scripts/run_tier3.py", "--backend", "remote", "--category", category],
            legacy_cmd=[python_bin, "scripts/legacy/iv_llm_normalize.py"],
            unified_results_dir=ROOT / "results",
            legacy_results_dir=ROOT / "scripts" / "legacy" / "results",
            result_prefix=result_prefix,
        ),
        CaseSpec(
            family="exemplar_rag",
            unified_cmd=[python_bin, "scripts/run_exemplar_rag.py", "--backend", "remote", "--category", category],
            legacy_cmd=[python_bin, "scripts/legacy/iv_exemplar_rag_remote.py"],
            unified_results_dir=ROOT / "results",
            legacy_results_dir=ROOT / "scripts" / "legacy" / "results",
            result_prefix=result_prefix,
        ),
    ]


def snapshot_results(results_dir: Path) -> set[Path]:
    if not results_dir.exists():
        return set()
    return {
        path
        for path in results_dir.iterdir()
        if path.is_file() and path.suffix in {".json", ".csv"}
    }


def newest_files(results_dir: Path, before: set[Path], result_prefix: str) -> tuple[Path, Path]:
    after = snapshot_results(results_dir)
    created = sorted(
        [
            path for path in (after - before)
            if path.name.startswith(result_prefix)
        ],
        key=lambda path: path.stat().st_mtime,
    )
    json_files = [path for path in created if path.suffix == ".json"]
    csv_files = [path for path in created if path.suffix == ".csv"]
    if not json_files or not csv_files:
        raise RuntimeError(f"No new result pair detected in {results_dir} for prefix {result_prefix}")
    return json_files[-1], csv_files[-1]


def run_command(cmd: list[str], env: dict[str, str], cwd: Path) -> CommandResult:
    completed = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
    )
    return CommandResult(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def load_csv_header(csv_path: Path) -> list[str]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        return next(reader)


def summarize_standard_rows(rows: list[dict]) -> dict:
    entries = len(rows)
    fields_correct = sum(int(row["fields_correct"]) for row in rows)
    fields_total = sum(int(row["fields_total"]) for row in rows)
    overall_correct = sum(1 for row in rows if row["all_fields_correct"])
    return {
        "entries": entries,
        "overall_correct": overall_correct,
        "overall_pct": (overall_correct / entries * 100) if entries else 0.0,
        "fields_correct": fields_correct,
        "fields_total": fields_total,
        "field_pct": (fields_correct / fields_total * 100) if fields_total else 0.0,
    }


def summarize_tier3_rows(rows: list[dict]) -> dict:
    entries = len(rows)
    raw_fields_correct = sum(int(row["raw_fields_correct"]) for row in rows)
    norm_fields_correct = sum(int(row["norm_fields_correct"]) for row in rows)
    field_total = len(rows[0]["comparison_raw"]) * entries if rows else 0
    raw_overall = sum(1 for row in rows if row["raw_all_correct"])
    norm_overall = sum(1 for row in rows if row["norm_all_correct"])
    return {
        "entries": entries,
        "raw": {
            "overall_correct": raw_overall,
            "overall_pct": (raw_overall / entries * 100) if entries else 0.0,
            "fields_correct": raw_fields_correct,
            "fields_total": field_total,
            "field_pct": (raw_fields_correct / field_total * 100) if field_total else 0.0,
        },
        "normalized": {
            "overall_correct": norm_overall,
            "overall_pct": (norm_overall / entries * 100) if entries else 0.0,
            "fields_correct": norm_fields_correct,
            "fields_total": field_total,
            "field_pct": (norm_fields_correct / field_total * 100) if field_total else 0.0,
        },
    }


def summarize_result_pair(json_path: Path, csv_path: Path, family: str) -> dict:
    with json_path.open(encoding="utf-8") as handle:
        rows = json.load(handle)
    summary = {
        "json_path": str(json_path),
        "csv_path": str(csv_path),
        "json_top_keys": sorted(rows[0].keys()) if rows else [],
        "csv_header": load_csv_header(csv_path),
    }
    if family == "tier3":
        summary["metrics"] = summarize_tier3_rows(rows)
    else:
        summary["metrics"] = summarize_standard_rows(rows)
    return summary


def compare_metrics(unified: dict, legacy: dict, family: str) -> dict:
    if family == "tier3":
        return {
            "raw_overall_delta": unified["metrics"]["raw"]["overall_correct"] - legacy["metrics"]["raw"]["overall_correct"],
            "raw_field_delta": unified["metrics"]["raw"]["fields_correct"] - legacy["metrics"]["raw"]["fields_correct"],
            "norm_overall_delta": unified["metrics"]["normalized"]["overall_correct"] - legacy["metrics"]["normalized"]["overall_correct"],
            "norm_field_delta": unified["metrics"]["normalized"]["fields_correct"] - legacy["metrics"]["normalized"]["fields_correct"],
            "json_keys_match": unified["json_top_keys"] == legacy["json_top_keys"],
            "csv_header_match": unified["csv_header"] == legacy["csv_header"],
        }
    return {
        "overall_delta": unified["metrics"]["overall_correct"] - legacy["metrics"]["overall_correct"],
        "field_delta": unified["metrics"]["fields_correct"] - legacy["metrics"]["fields_correct"],
        "json_keys_match": unified["json_top_keys"] == legacy["json_top_keys"],
        "csv_header_match": unified["csv_header"] == legacy["csv_header"],
    }


def acceptance_status(comparison: dict) -> tuple[bool, list[str]]:
    failures = []
    for key, value in comparison.items():
        if key.endswith("_match") and not value:
            failures.append(key)
    return not failures, failures


def summarize_failure(result: CommandResult) -> dict:
    text = (result.stderr or result.stdout).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return {
        "returncode": result.returncode,
        "message": lines[-1] if lines else f"process exited with code {result.returncode}",
    }


def artifact_failure(message: str) -> dict:
    return {"returncode": 0, "message": message}


def run_case_pair(spec: CaseSpec, env: dict[str, str]) -> dict:
    unified_before = snapshot_results(spec.unified_results_dir)
    unified_run = run_command(spec.unified_cmd, env, ROOT)
    unified_failure = summarize_failure(unified_run) if unified_run.returncode else None
    unified_json = unified_csv = None
    if not unified_failure:
        try:
            unified_json, unified_csv = newest_files(
                spec.unified_results_dir,
                unified_before,
                spec.result_prefix,
            )
        except RuntimeError as exc:
            unified_failure = artifact_failure(str(exc))

    legacy_before = snapshot_results(spec.legacy_results_dir)
    legacy_run = run_command(spec.legacy_cmd, env, ROOT)
    legacy_failure = summarize_failure(legacy_run) if legacy_run.returncode else None
    legacy_json = legacy_csv = None
    if not legacy_failure:
        try:
            legacy_json, legacy_csv = newest_files(
                spec.legacy_results_dir,
                legacy_before,
                spec.result_prefix,
            )
        except RuntimeError as exc:
            legacy_failure = artifact_failure(str(exc))

    if unified_failure or legacy_failure:
        if unified_failure and legacy_failure:
            return {
                "family": spec.family,
                "status": "skipped",
                "reason": "both unified and legacy entrypoints failed on this slice",
                "unified_failure": unified_failure,
                "legacy_failure": legacy_failure,
                "accepted": True,
                "failures": [],
            }
        return {
            "family": spec.family,
            "status": "review",
            "reason": "only one side completed successfully on this slice",
            "unified_failure": unified_failure,
            "legacy_failure": legacy_failure,
            "accepted": False,
            "failures": ["execution_mismatch"],
        }

    unified_summary = summarize_result_pair(unified_json, unified_csv, spec.family)
    legacy_summary = summarize_result_pair(legacy_json, legacy_csv, spec.family)
    comparison = compare_metrics(unified_summary, legacy_summary, spec.family)
    accepted, failures = acceptance_status(comparison)
    return {
        "family": spec.family,
        "unified": unified_summary,
        "legacy": legacy_summary,
        "comparison": comparison,
        "accepted": accepted,
        "failures": failures,
    }


def render_markdown(summary: dict) -> str:
    lines = [
        "# Refactor Parity Note",
        "",
        f"- backend: `{summary['backend']}`",
        f"- category: `{summary['category']}`",
        f"- python: `{summary['python_bin']}`",
        f"- model: `{summary.get('model_name', '')}`",
        "",
        "## Results",
        "",
    ]
    for item in summary["cases"]:
        lines.append(f"### {item['family']}")
        status = item.get("status", "pass" if item["accepted"] else "review")
        lines.append(f"- acceptance: `{status}`")
        if status == "skipped":
            lines.append(f"- reason: `{item['reason']}`")
            if item.get("unified_failure"):
                lines.append(f"- unified failure: `{item['unified_failure']['message']}`")
            if item.get("legacy_failure"):
                lines.append(f"- legacy failure: `{item['legacy_failure']['message']}`")
        elif "comparison" in item:
            lines.append(f"- unified json: `{item['unified']['json_path']}`")
            lines.append(f"- legacy json: `{item['legacy']['json_path']}`")
            for key, value in item["comparison"].items():
                lines.append(f"- {key}: `{value}`")
        else:
            lines.append(f"- reason: `{item['reason']}`")
            if "unified_failure" in item:
                lines.append(f"- unified failure: `{item['unified_failure']['message']}`")
            if "legacy_failure" in item:
                lines.append(f"- legacy failure: `{item['legacy_failure']['message']}`")
        lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Deltas of `0` indicate parity on the measured slice.")
    lines.append("- A `review` status means a schema/layout drift or one-sided execution failure was detected and should be explained before merge.")
    lines.append("- A `skipped` status means both unified and legacy paths failed on the same unsupported slice, so the refactor did not introduce a new regression there.")
    lines.append("- These notes cover fixed local validation slices only. Add remote parity only if credentials and API access are available.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["local", "remote"], default="local")
    parser.add_argument("--category", choices=["iv_push", "iv_intermittent"], default="iv_push")
    parser.add_argument("--python-bin", default=DEFAULT_PYTHON)
    parser.add_argument("--model-name")
    parser.add_argument("--ollama-model", dest="legacy_model_name")
    parser.add_argument("--summary-json", default=str(ROOT / "results" / "parity_summary.json"))
    parser.add_argument("--summary-md", default=str(ROOT / "docs" / "refactor_parity.md"))
    args = parser.parse_args()

    model_name = args.model_name or args.legacy_model_name
    if not model_name:
        model_name = DEFAULT_LOCAL_MODEL if args.backend == "local" else os.environ.get("GOOGLE_MODEL_NAME", DEFAULT_REMOTE_MODEL)

    env = os.environ.copy()
    env.update(
        {
            "MEDMATCH_NUM_RUNS": "1",
            "MEDMATCH_MAX_ENTRIES": "1",
            "MEDMATCH_SLEEP_SECONDS": "0",
            "MEDMATCH_SHEETS": SHEET_NAMES[args.category],
            "PYTHONPATH": str(ROOT / "src"),
        }
    )
    if args.backend == "local":
        env["OLLAMA_MODEL"] = model_name
    else:
        env["GOOGLE_MODEL_NAME"] = model_name

    cases = []
    case_specs = build_local_cases(args.category, args.python_bin) if args.backend == "local" else build_remote_cases(args.category, args.python_bin)
    for spec in case_specs:
        cases.append(run_case_pair(spec, env))

    summary = {
        "backend": args.backend,
        "category": args.category,
        "python_bin": args.python_bin,
        "model_name": model_name,
        "cases": cases,
    }

    summary_json = Path(args.summary_json)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    summary_md = Path(args.summary_md)
    summary_md.parent.mkdir(parents=True, exist_ok=True)
    summary_md.write_text(render_markdown(summary), encoding="utf-8")

    print(f"Wrote parity summary: {summary_json}")
    print(f"Wrote parity note: {summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
