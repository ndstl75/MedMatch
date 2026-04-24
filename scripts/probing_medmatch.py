"""
Canonical MedMatch runner.

This keeps the lab CSV-backed runner as the main entrypoint for baseline
prompting and extends it with CoT and normalization modes so we do not carry
parallel runner loops in `src/medmatch/experiments/`.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from typing import Any, Dict, Iterable, List, Optional


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC_ROOT = os.path.join(_REPO_ROOT, "src")
for candidate in (_REPO_ROOT, _SRC_ROOT):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from prompt_medmatch import (
    SYSTEM_PROMPT,
    build_cot_extract_prompt,
    build_cot_reason_prompt,
    build_iv_continuous_messages_one_shot,
    build_iv_continuous_messages_two_shot_multi_turn,
    build_iv_continuous_messages_zero_shot,
    build_iv_intermit_messages_one_shot,
    build_iv_intermit_messages_one_shot_multi_turn,
    build_iv_intermit_messages_zero_shot,
    build_iv_push_messages_one_shot,
    build_iv_push_messages_one_shot_multi_turn,
    build_iv_push_messages_zero_shot,
    build_local_normalization_prompt,
    build_po_liquid_messages_one_shot,
    build_po_liquid_messages_one_shot_multi_turn,
    build_po_liquid_messages_zero_shot,
    build_po_solid_messages_one_shot,
    build_po_solid_messages_one_shot_multi_turn,
    build_po_solid_messages_zero_shot,
    build_remote_normalization_oral_instruction,
    build_remote_normalization_prompt,
    get_cot_reason_system_prompt,
)
from medmatch.core.schema import (
    BASELINE_SHEET_CONFIG,
    LOCAL_NORMALIZATION_IV_SHEET_CONFIG,
    LOCAL_NORMALIZATION_ORAL_SHEET_CONFIG,
)
from medmatch.core.paths import (
    SCORER_VERSION,
    current_results_root,
    dataset_version_for_path,
    default_data_dir,
)
from medmatch.core.scorer import (
    all_fields_match,
    coerce_output_object,
    compare_results,
    normalize_strict,
    parse_json_response,
)
from medmatch.llm.config import SUPPORTED_BACKENDS, canonical_backend_name
from medmatch.llm.local_ollama import LocalOllamaBackend
from medmatch.llm.remote_api import AzureOpenAIBackend, LocalQwenOpenAIBackend, OpenAICompatibleBackend
from medmatch.llm.remote_gemma import RemoteGemmaBackend

try:
    from vllm import LLM, SamplingParams

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LLM = None
    SamplingParams = None

try:
    from transformers import AutoTokenizer

    HF_TOKENIZER_AVAILABLE = True
except ImportError:
    HF_TOKENIZER_AVAILABLE = False
    AutoTokenizer = None


PROMPTING_CHOICES = ("zero", "few", "one_shot", "cot", "normalization")
MODE_CHOICES = tuple(SUPPORTED_BACKENDS) + ("vllm",)
REMOTE_STYLE_MODES = {"openai", "azure", "remote", "google", "qwen_local"}
DATASET_ORDER = ["po_solid", "po_liquid", "iv_intermit", "iv_push", "iv_continuous"]
OUTPUT_SUBDIRS = {
    "zero": "zero-shot",
    "few": "few-shot",
    "one_shot": "one-shot",
    "cot": "cot",
    "normalization": "normalization",
}
DEFAULT_DATA_DIR = default_data_dir()
DEFAULT_RESULTS_ROOT = current_results_root()


DATASET_SPECS = {
    "po_solid": {
        "sheet_name": "PO Solid (40)",
        "filename": "med_match - po_solid.csv",
        "prompt_text_col": 2,
        "ground_truth_text_col": 1,
        "family": "oral",
    },
    "po_liquid": {
        "sheet_name": "PO liquid (10)",
        "filename": "med_match - po_liquid.csv",
        "prompt_text_col": 2,
        "ground_truth_text_col": 1,
        "family": "oral",
    },
    "iv_intermit": {
        "sheet_name": "IV intermittent (16)",
        "filename": "med_match - iv_i.csv",
        "prompt_text_col": 2,
        "ground_truth_text_col": 1,
        "family": "iv",
    },
    "iv_push": {
        "sheet_name": "IV push (17)",
        "filename": "med_match - iv_p.csv",
        "prompt_text_col": 2,
        "ground_truth_text_col": 1,
        "family": "iv",
    },
    "iv_continuous": {
        "sheet_name": "IV continuous (16)",
        "filename": "med_match - iv_c.csv",
        "prompt_text_col": 2,
        "ground_truth_text_col": 1,
        "family": "iv",
    },
}


BASELINE_BUILDERS = {
    "zero": {
        "PO Solid (40)": build_po_solid_messages_zero_shot,
        "PO liquid (10)": build_po_liquid_messages_zero_shot,
        "IV intermittent (16)": build_iv_intermit_messages_zero_shot,
        "IV push (17)": build_iv_push_messages_zero_shot,
        "IV continuous (16)": build_iv_continuous_messages_zero_shot,
    },
    "few": {
        "PO Solid (40)": build_po_solid_messages_one_shot_multi_turn,
        "PO liquid (10)": build_po_liquid_messages_one_shot_multi_turn,
        "IV intermittent (16)": build_iv_intermit_messages_one_shot_multi_turn,
        "IV push (17)": build_iv_push_messages_one_shot_multi_turn,
        "IV continuous (16)": build_iv_continuous_messages_two_shot_multi_turn,
    },
    "one_shot": {
        "PO Solid (40)": build_po_solid_messages_one_shot,
        "PO liquid (10)": build_po_liquid_messages_one_shot,
        "IV intermittent (16)": build_iv_intermit_messages_one_shot,
        "IV push (17)": build_iv_push_messages_one_shot,
        "IV continuous (16)": build_iv_continuous_messages_one_shot,
    },
}


def sanitize_model_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace(" ", "_")


def default_model_name_for_mode(mode: str) -> str:
    normalized_mode = canonical_backend_name(mode)
    if normalized_mode == "local":
        return os.environ.get("OLLAMA_MODEL", "gemma4:e4b")
    if normalized_mode in {"remote", "google"}:
        return os.environ.get("GOOGLE_MODEL_NAME", "gemma-3-27b-it")
    if normalized_mode == "azure":
        return os.environ.get("AZURE_OPENAI_DEPLOYMENT") or os.environ.get("AZURE_MODEL_NAME", "gpt-4o-mini")
    if normalized_mode == "qwen_local":
        return (
            os.environ.get("LOCAL_OPENAI_MODEL_NAME")
            or os.environ.get("OPENAI_MODEL")
            or os.environ.get("OPENAI_MODEL_NAME")
            or "Qwen/Qwen3.6-35B-A3B"
        )
    return os.environ.get("OPENAI_MODEL_NAME") or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"


def sheet_safe_name(sheet_name: str) -> str:
    return sheet_name.replace(" ", "_").replace("(", "").replace(")", "")


def is_remote_style_mode(mode: str) -> bool:
    return mode in REMOTE_STYLE_MODES


def messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    parts = []
    for msg in messages:
        role = msg.get("role", "user").capitalize()
        parts.append(f"{role}: {msg.get('content', '')}")
    parts.append("Assistant:")
    return "\n".join(parts)


def _read_cell(raw: List[str], index: int) -> str:
    if index < 0 or index >= len(raw):
        return ""
    value = raw[index]
    return "" if value is None else str(value).strip()


def _sample_rows(rows: List[Dict[str, Any]], subset_size: Optional[int]) -> List[Dict[str, Any]]:
    if subset_size and subset_size < len(rows):
        rng = random.Random(42)
        return rng.sample(rows, subset_size)
    return rows


def load_csv_rows(data_dir: str, dataset_key: str, subset_size: Optional[int]) -> List[Dict[str, Any]]:
    spec = DATASET_SPECS[dataset_key]
    sheet_name = spec["sheet_name"]
    sheet_config = BASELINE_SHEET_CONFIG[sheet_name]
    path = os.path.join(data_dir, spec["filename"])

    rows: List[Dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for raw in reader:
            prompt = _read_cell(raw, spec["prompt_text_col"])
            if not prompt:
                break
            ground_truth = {
                key: _read_cell(raw, col_idx - 1)
                for key, col_idx in sheet_config["ground_truth_cols"].items()
            }
            rows.append(
                {
                    "dataset_key": dataset_key,
                    "sheet_name": sheet_name,
                    "medication": _read_cell(raw, 0),
                    "prompt": prompt,
                    "ground_truth_text": _read_cell(raw, spec["ground_truth_text_col"]),
                    "ground_truth": ground_truth,
                }
            )
    return _sample_rows(rows, subset_size)


def build_datasets(data_dir: str, prompting_type: str, subset_size: Optional[int], dataset_keys: Iterable[str]) -> Dict[str, List[Dict[str, Any]]]:
    del prompting_type
    return {
        DATASET_SPECS[key]["sheet_name"]: load_csv_rows(data_dir, key, subset_size)
        for key in dataset_keys
    }


def build_baseline_messages(sheet_name: str, prompting_type: str, medication_prompt: str) -> List[Dict[str, str]]:
    return BASELINE_BUILDERS[prompting_type][sheet_name](medication_prompt)


def create_backend(mode: str, model_name: str, temperature: float):
    mode = canonical_backend_name(mode)
    selected_model = model_name
    if mode in {"local", "remote", "google"} and model_name == "gpt-4o-mini":
        selected_model = None
    if mode == "local":
        return LocalOllamaBackend(model=selected_model, temperature=temperature)
    if mode == "qwen_local":
        return LocalQwenOpenAIBackend(model=selected_model, temperature=temperature)
    if mode == "openai":
        return OpenAICompatibleBackend(model=selected_model, temperature=temperature)
    if mode == "azure":
        return AzureOpenAIBackend(model=selected_model, temperature=temperature)
    return RemoteGemmaBackend(model=selected_model, temperature=temperature)


def init_runtime(args) -> Dict[str, Any]:
    if args.mode == "vllm":
        if not VLLM_AVAILABLE:
            raise ImportError("vllm package not available.")
        runtime: Dict[str, Any] = {
            "kind": "vllm",
            "llm": LLM(
                model=args.model_name,
                tensor_parallel_size=max(1, args.number_gpus),
                trust_remote_code=True,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=args.max_model_len,
            ),
            "sampling_params": SamplingParams(
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_new_tokens,
            ),
            "tokenizer": None,
        }
        if "qwen" in args.model_name.lower():
            if not HF_TOKENIZER_AVAILABLE:
                raise RuntimeError(
                    "Qwen models require transformers tokenizer to disable thinking. "
                    "Install transformers or choose a non-Qwen model."
                )
            runtime["tokenizer"] = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        return runtime

    return {
        "kind": "backend",
        "backend": create_backend(args.mode, args.model_name, args.temperature),
        "temperature": args.temperature,
    }


def render_messages_for_runtime(runtime: Dict[str, Any], messages: List[Dict[str, str]]) -> str:
    if runtime["kind"] == "vllm":
        tokenizer = runtime.get("tokenizer")
        if tokenizer is not None:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
    return messages_to_prompt(messages)


def generate_text_from_messages(runtime: Dict[str, Any], messages: List[Dict[str, str]]) -> str:
    if runtime["kind"] == "vllm":
        prompt = render_messages_for_runtime(runtime, messages)
        outputs = runtime["llm"].generate([prompt], sampling_params=runtime["sampling_params"])
        return outputs[0].outputs[0].text.strip()

    backend = runtime["backend"]
    if len(messages) == 2 and messages[0].get("role") == "system" and messages[1].get("role") == "user":
        return backend.generate_text(messages[0]["content"], messages[1]["content"], temperature=runtime["temperature"])
    return backend.generate_text("", messages_to_prompt(messages), temperature=runtime["temperature"])


def generate_text(runtime: Dict[str, Any], system_prompt: str, user_prompt: str) -> str:
    if runtime["kind"] == "vllm":
        return generate_text_from_messages(
            runtime,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
    return runtime["backend"].generate_text(system_prompt, user_prompt, temperature=runtime["temperature"])


def generate_json(runtime: Dict[str, Any], system_prompt: str, user_prompt: str, expected_keys: List[str]) -> tuple[Dict[str, Any], str]:
    text = generate_text(runtime, system_prompt, user_prompt)
    parsed = parse_json_response(text)
    return coerce_output_object(parsed, expected_keys), text


def select_dataset_keys(prompting_type: str, dataset_keys: Optional[Iterable[str]] = None) -> List[str]:
    requested = list(dataset_keys or [])
    if prompting_type in {"zero", "few", "one_shot"}:
        allowed = DATASET_ORDER
    elif prompting_type == "cot":
        allowed = ["iv_intermit", "iv_push", "iv_continuous"]
    elif prompting_type == "normalization":
        allowed = ["po_solid", "po_liquid", "iv_intermit", "iv_push", "iv_continuous"]
    else:
        raise ValueError(f"Unsupported prompting_type: {prompting_type}")

    if not requested:
        return allowed

    invalid = [key for key in requested if key not in allowed]
    if invalid:
        raise ValueError(
            f"Prompting type '{prompting_type}' does not support datasets: {', '.join(invalid)}"
        )
    return [key for key in allowed if key in requested]


def write_record(handle, record: Dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_run_metadata(
    output_dir: str,
    *,
    mode: str,
    model_name: str,
    prompting_type: str,
    num_runs: int,
    data_dir: str,
    dataset_keys: List[str],
    datasets: Dict[str, List[Dict[str, Any]]],
) -> None:
    dataset_version = dataset_version_for_path(data_dir)
    metadata = {
        "dataset_version": dataset_version,
        "dataset_dir": os.path.abspath(data_dir),
        "scorer_version": SCORER_VERSION,
        "prompting_type": prompting_type,
        "mode": mode,
        "model_name": model_name,
        "num_runs": num_runs,
        "dataset_keys": dataset_keys,
        "row_counts_by_dataset": {
            dataset_key: len(datasets[DATASET_SPECS[dataset_key]["sheet_name"]])
            for dataset_key in dataset_keys
        },
        "comparison_notice": (
            f"Do not compare {dataset_version} outputs directly against results computed on "
            "other MedMatch dataset versions without explicit dataset-version disclosure."
        ),
    }
    metadata_path = os.path.join(output_dir, f"{sanitize_model_name(model_name)}_run_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def titration_fields_present(ground_truth: Dict[str, Any]) -> bool:
    fields = [
        "titration dose",
        "titration unit of measure",
        "titration frequency",
        "titration goal based on physiologic response, laboratory result, or assessment score",
    ]
    return any(str(ground_truth.get(field, "")).strip() for field in fields)


def get_local_normalization_resources(family: str):
    if family == "oral":
        return LOCAL_NORMALIZATION_ORAL_SHEET_CONFIG
    return LOCAL_NORMALIZATION_IV_SHEET_CONFIG


def build_normalization_extract_prompt(instruction: str, prompt: str, expected_keys: List[str]) -> str:
    return (
        f"{instruction}\n\n"
        "Return one JSON object only.\n"
        "Do not wrap the JSON in markdown.\n"
        f"Use exactly these keys in this order: {', '.join(expected_keys)}.\n\n"
        f"Now process this medication order:\n{prompt}"
    )


def process_baseline_entry(runtime: Dict[str, Any], prompting_type: str, row: Dict[str, Any], model_name: str, run_id: int) -> Dict[str, Any]:
    messages = build_baseline_messages(row["sheet_name"], prompting_type, row["prompt"])
    response = generate_text_from_messages(runtime, messages)
    return {
        "dataset": row["dataset_key"],
        "run": run_id,
        "model": model_name,
        "medication": row["medication"],
        "prompt": row["prompt"],
        "ground_truth": row["ground_truth_text"],
        "response": response,
    }


def process_cot_entry(runtime: Dict[str, Any], mode: str, row: Dict[str, Any], run_id: int) -> Dict[str, Any]:
    sheet_name = row["sheet_name"]
    spec = BASELINE_SHEET_CONFIG[sheet_name]
    expected_keys = list(spec["ground_truth_cols"].keys())
    remote_mode = is_remote_style_mode(mode)

    reasoning = generate_text(
        runtime,
        get_cot_reason_system_prompt(),
        build_cot_reason_prompt(sheet_name, row["prompt"], remote_mode=remote_mode),
    )
    extract_prompt = build_cot_extract_prompt(
        sheet_name,
        reasoning,
        row["prompt"],
        spec["instruction"],
        expected_keys,
        remote_mode=remote_mode,
    )
    llm_output, raw_extract = generate_json(
        runtime,
        SYSTEM_PROMPT,
        extract_prompt,
        expected_keys,
    )
    comparison = compare_results(llm_output, row["ground_truth"], normalizer=normalize_strict)
    fields_correct = sum(1 for value in comparison.values() if value["match"])
    entry_correct = all_fields_match(comparison)
    entry_type = None
    if sheet_name == "IV continuous (16)":
        entry_type = "titratable" if titration_fields_present(row["ground_truth"]) else "non-titratable"

    return {
        "run": run_id,
        "medication": row["medication"],
        "prompt": row["prompt"],
        "ground_truth": row["ground_truth"],
        "reasoning": reasoning,
        "llm_output": llm_output,
        "raw_extract_response": raw_extract,
        "comparison": comparison,
        "fields_correct": fields_correct,
        "fields_total": len(expected_keys),
        "all_fields_correct": entry_correct,
        "entry_type": entry_type,
    }


def process_normalization_entry(runtime: Dict[str, Any], mode: str, row: Dict[str, Any], run_id: int) -> Dict[str, Any]:
    sheet_name = row["sheet_name"]
    family = DATASET_SPECS[row["dataset_key"]]["family"]
    remote_mode = is_remote_style_mode(mode)

    if remote_mode:
        instruction = (
            build_remote_normalization_oral_instruction(sheet_name)
            if family == "oral"
            else BASELINE_SHEET_CONFIG[sheet_name]["instruction"]
        )
    else:
        sheet_config = get_local_normalization_resources(family)
        instruction = sheet_config[sheet_name]["instruction"]

    expected_keys = list(BASELINE_SHEET_CONFIG[sheet_name]["ground_truth_cols"].keys())
    extract_prompt = build_normalization_extract_prompt(instruction, row["prompt"], expected_keys)
    raw_obj, raw_text = generate_json(
        runtime,
        SYSTEM_PROMPT,
        extract_prompt,
        expected_keys,
    )

    raw_json = json.dumps(raw_obj, indent=2, default=str)
    normalize_prompt = (
        build_remote_normalization_prompt(
            row["prompt"],
            raw_json,
            family=family,
            sheet_name=sheet_name,
        )
        if remote_mode
        else build_local_normalization_prompt(row["prompt"], raw_json, family=family, sheet_name=sheet_name)
    )
    normalize_text = generate_text(runtime, SYSTEM_PROMPT, normalize_prompt)
    parsed = parse_json_response(normalize_text)
    if isinstance(parsed, dict):
        normalized_obj = {key: parsed.get(key, raw_obj.get(key, "")) for key in expected_keys}
    else:
        normalized_obj = dict(raw_obj)

    comparison_raw = compare_results(raw_obj, row["ground_truth"], normalizer=normalize_strict)
    comparison_normalized = compare_results(normalized_obj, row["ground_truth"], normalizer=normalize_strict)
    raw_fields_correct = sum(1 for value in comparison_raw.values() if value["match"])
    norm_fields_correct = sum(1 for value in comparison_normalized.values() if value["match"])

    return {
        "run": run_id,
        "medication": row["medication"],
        "prompt": row["prompt"],
        "ground_truth": row["ground_truth"],
        "raw_output": raw_obj,
        "raw_response": raw_text,
        "normalized_output": normalized_obj,
        "normalized_response": normalize_text,
        "comparison_raw": comparison_raw,
        "comparison_normalized": comparison_normalized,
        "raw_fields_correct": raw_fields_correct,
        "norm_fields_correct": norm_fields_correct,
        "raw_all_correct": all_fields_match(comparison_raw),
        "norm_all_correct": all_fields_match(comparison_normalized),
    }


def run_medmatch_pipeline(
    *,
    mode: str,
    model_name: Optional[str],
    prompting_type: str,
    num_runs: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    data_dir: str,
    output_dir: Optional[str],
    subset_size: Optional[int],
    dataset_keys: Optional[Iterable[str]] = None,
    batch_size: int = 10,
    number_gpus: int = 2,
    gpu_memory_utilization: float = 0.85,
    max_model_len: int = 4096,
) -> Dict[str, List[Dict[str, Any]]]:
    del batch_size
    resolved_model_name = model_name or default_model_name_for_mode(mode)

    runtime_args = argparse.Namespace(
        mode=mode,
        model_name=resolved_model_name,
        temperature=temperature,
        number_gpus=number_gpus,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
    )
    runtime = init_runtime(runtime_args)
    selected_keys = select_dataset_keys(prompting_type, dataset_keys)
    datasets = build_datasets(data_dir, prompting_type, subset_size, selected_keys)
    results_by_sheet = {sheet_name: [] for sheet_name in datasets}

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for run_id in range(1, num_runs + 1):
        output_path = None
        output_handle = None
        if output_dir:
            output_path = os.path.join(output_dir, f"{sanitize_model_name(resolved_model_name)}_run{run_id}.jsonl")
            output_handle = open(output_path, "w", encoding="utf-8")

        try:
            for dataset_key in selected_keys:
                sheet_name = DATASET_SPECS[dataset_key]["sheet_name"]
                for row in datasets[sheet_name]:
                    if prompting_type in {"zero", "few", "one_shot"}:
                        record = process_baseline_entry(runtime, prompting_type, row, resolved_model_name, run_id)
                    elif prompting_type == "cot":
                        record = process_cot_entry(runtime, mode, row, run_id)
                    else:
                        record = process_normalization_entry(runtime, mode, row, run_id)

                    results_by_sheet[sheet_name].append(record)
                    if output_handle is not None:
                        write_record(output_handle, record)
        finally:
            if output_handle is not None:
                output_handle.close()

    if output_dir:
        write_run_metadata(
            output_dir,
            mode=mode,
            model_name=resolved_model_name,
            prompting_type=prompting_type,
            num_runs=num_runs,
            data_dir=data_dir,
            dataset_keys=selected_keys,
            datasets=datasets,
        )

    return results_by_sheet


def build_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MedMatch prompting against the lab CSV datasets.")
    parser.add_argument(
        "--mode",
        choices=MODE_CHOICES,
        default="openai",
        help="openai | azure | local | remote | google | qwen_local | vllm.",
    )
    parser.add_argument("--model_name")
    parser.add_argument(
        "--prompting_type",
        choices=PROMPTING_CHOICES,
        default="zero",
        help="zero | few | one_shot | cot | normalization",
    )
    parser.add_argument("--num_runs", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument(
        "--data_dir",
        default=DEFAULT_DATA_DIR,
    )
    parser.add_argument("--output_dir", default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--subset_size", type=int, default=None, help="Optional subset size for quick tests.")
    parser.add_argument("--batch_size", type=int, default=10, help="Reserved for compatibility.")
    parser.add_argument("--number_gpus", type=int, default=2, help="Tensor parallel size for vLLM.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85, help="vLLM GPU memory utilization fraction.")
    parser.add_argument("--max_model_len", type=int, default=4096, help="vLLM maximum sequence length.")
    return parser


def main() -> None:
    parser = build_args()
    args = parser.parse_args()
    output_dir = os.path.join(args.output_dir, OUTPUT_SUBDIRS[args.prompting_type])

    run_medmatch_pipeline(
        mode=args.mode,
        model_name=args.model_name,
        prompting_type=args.prompting_type,
        num_runs=args.num_runs,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        data_dir=args.data_dir,
        output_dir=output_dir,
        subset_size=args.subset_size,
        dataset_keys=None,
        batch_size=args.batch_size,
        number_gpus=args.number_gpus,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )


if __name__ == "__main__":
    main()
