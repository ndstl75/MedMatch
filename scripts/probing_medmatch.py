"""
MedMatch prompting runner.

Loads MedMatch CSVs (PO, IV intermittent, IV push, IV continuous), builds
formatting prompts, and queries the selected LLM. Outputs JSONL files per
model/run with prompts, ground truth, and model responses.

Supports zero-shot, few-shot, and single-turn one-shot prompting. Results are saved in separate directories:
- ./results/med_match/zero-shot/ for zero-shot prompting
- ./results/med_match/few-shot/ for few-shot/multi-turn prompting
- ./results/med_match/one-shot/ for single-turn one-shot prompting

Files are named as: {model_name}_run{run_id}.jsonl

Examples (from repository root; see README.md for environment variables):

    python scripts/probing_medmatch.py --mode openai --model_name gpt-4o-mini \\
        --prompting_type zero --num_runs 3 --temperature 0.7 --batch_size 10

    CUDA_VISIBLE_DEVICES=0,1 python scripts/probing_medmatch.py --mode vllm \\
        --model_name google/medgemma-27b-text-it --prompting_type zero \\
        --num_runs 3 --batch_size 50

    python scripts/probing_medmatch.py --mode azure --model_name azure-gpt-5-chat \\
        --prompting_type one_shot --num_runs 3 --temperature 1.2 --batch_size 5
"""

import argparse
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

# Repository root (parent of scripts/) on sys.path for `import src`
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _REPO_ROOT)

import pandas as pd
from tqdm import tqdm

# Optional: load environment variables from .env if available
try:
    from dotenv import load_dotenv, find_dotenv

    load_dotenv(find_dotenv(), override=True)
except ImportError:
    pass

from src.prompt_medmatch import (
    build_po_solid_messages_zero_shot,
    build_po_solid_messages_one_shot,
    build_po_solid_messages_one_shot_multi_turn,
    build_po_liquid_messages_zero_shot,
    build_po_liquid_messages_one_shot,
    build_po_liquid_messages_one_shot_multi_turn,
    build_iv_intermit_messages_zero_shot,
    build_iv_intermit_messages_one_shot,
    build_iv_intermit_messages_one_shot_multi_turn,
    build_iv_push_messages_zero_shot,
    build_iv_push_messages_one_shot,
    build_iv_push_messages_one_shot_multi_turn,
    build_iv_continuous_messages_zero_shot,
    build_iv_continuous_messages_one_shot,
    build_iv_continuous_messages_two_shot_multi_turn,
)

# Optional providers
try:
    from openai import OpenAI, AzureOpenAI

    OPENAI_AVAILABLE = True
    AZURE_OPENAI_AVAILABLE = True
except ImportError:
    try:
        from openai import OpenAI

        OPENAI_AVAILABLE = True
        AZURE_OPENAI_AVAILABLE = False
    except ImportError:
        OPENAI_AVAILABLE = False
        AZURE_OPENAI_AVAILABLE = False

try:
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider

    AZURE_IDENTITY_AVAILABLE = True
except ImportError:
    AZURE_IDENTITY_AVAILABLE = False

try:
    from vllm import LLM, SamplingParams

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

try:
    from transformers import AutoTokenizer
    HF_TOKENIZER_AVAILABLE = True
except ImportError:
    HF_TOKENIZER_AVAILABLE = False

def sanitize_model_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace(" ", "_")


def messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    """
    Flatten chat messages into a simple prompt for vLLM generation.
    """
    parts = []
    for msg in messages:
        role = msg.get("role", "user").capitalize()
        parts.append(f"{role}: {msg.get('content', '')}")
    parts.append("Assistant:")
    return "\n".join(parts)


def load_dataframe(path: str, subset_size: Optional[int]) -> pd.DataFrame:
    df = pd.read_csv(path)
    if subset_size and subset_size < len(df):
        return df.sample(n=subset_size, random_state=42).reset_index(drop=True)
    return df


def build_datasets(base_dir: str, prompting_type: str) -> Dict[str, Dict]:
    """Configure datasets with paths and prompt builders."""
    # Choose builders based on prompting type
    if prompting_type == "zero":
        po_solid_builder = build_po_solid_messages_zero_shot
        po_liquid_builder = build_po_liquid_messages_zero_shot
        iv_intermit_builder = build_iv_intermit_messages_zero_shot
        iv_push_builder = build_iv_push_messages_zero_shot
        iv_continuous_builder = build_iv_continuous_messages_zero_shot
    elif prompting_type == "few":
        po_solid_builder = build_po_solid_messages_one_shot_multi_turn
        po_liquid_builder = build_po_liquid_messages_one_shot_multi_turn
        iv_intermit_builder = build_iv_intermit_messages_one_shot_multi_turn
        iv_push_builder = build_iv_push_messages_one_shot_multi_turn
        iv_continuous_builder = build_iv_continuous_messages_two_shot_multi_turn
    elif prompting_type == "one_shot":
        # Single-turn one-shot prompting for all datasets
        po_solid_builder = build_po_solid_messages_one_shot
        po_liquid_builder = build_po_liquid_messages_one_shot
        iv_intermit_builder = build_iv_intermit_messages_one_shot
        iv_push_builder = build_iv_push_messages_one_shot
        iv_continuous_builder = build_iv_continuous_messages_one_shot
    else:
        raise ValueError(f"Invalid prompting_type: {prompting_type}. Must be 'zero', 'few', or 'one_shot'.")

    return {
        "po_solid": {
            "path": os.path.join(base_dir, "med_match - po_solid.csv"),
            "prompt_col": "Medication prompt (sentence format)",
            "gt_col": "Medication JSON (ground truth)",
            "builder": po_solid_builder,
        },
        "po_liquid": {
            "path": os.path.join(base_dir, "med_match - po_liquid.csv"),
            "prompt_col": "Medication prompt (sentence format)",
            "gt_col": "Medication JSON (ground truth)",
            "builder": po_liquid_builder,
        },
        "iv_intermit": {
            "path": os.path.join(base_dir, "med_match - iv_i.csv"),
            "prompt_col": "Medication prompt (sentence format)",
            "gt_col": "Medication JSON",
            "builder": iv_intermit_builder,
        },
        "iv_push": {
            "path": os.path.join(base_dir, "med_match - iv_p.csv"),
            "prompt_col": "Medication prompt (sentence format)",
            "gt_col": "Medication JSON",
            "builder": iv_push_builder,
        },
        "iv_continuous": {
            "path": os.path.join(base_dir, "med_match - iv_c.csv"),
            "prompt_col": "Medication prompt (sentence format)",
            "gt_col": "Medication JSON",
            "builder": iv_continuous_builder,
        },
    }


def get_openai_client():
    if not OPENAI_AVAILABLE:
        raise ImportError("openai package not available.")
    if not os.getenv("OPENAI_API_KEY"):
        try:
            from dotenv import load_dotenv, find_dotenv
            load_dotenv(find_dotenv(), override=True)
        except ImportError:
            pass
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY missing for OpenAI mode (.env).")
    return OpenAI()


def get_azure_client(model_name: str) -> Tuple[Any, str]:
    """
    Create Azure OpenAI client and deployment name.
    model_name: e.g. 'azure-gpt-5-chat' or 'gpt-5-chat' (with --mode azure).
    Returns (client, deployment_name).
    """
    if not AZURE_OPENAI_AVAILABLE or not AZURE_IDENTITY_AVAILABLE:
        raise ImportError(
            "Azure mode requires: pip install azure-identity openai"
        )
    if not model_name.startswith("azure-"):
        model_name = f"azure-{model_name}"
    deployment = model_name.replace("azure-", "")
    endpoint = (os.getenv("AZURE_OPENAI_ENDPOINT") or "").strip().rstrip("/")
    if not endpoint:
        raise EnvironmentError(
            "Set AZURE_OPENAI_ENDPOINT to your Azure OpenAI HTTPS endpoint "
            "(Azure Portal: resource URL, e.g. https://<resource-name>.openai.azure.com)"
        )
    credential = DefaultAzureCredential(exclude_managed_identity_credential=True)
    token_provider = get_bearer_token_provider(
        credential, "https://cognitiveservices.azure.com/.default"
    )
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        azure_ad_token_provider=token_provider,
        api_version="2024-02-15-preview",
    )
    return client, deployment


def init_vllm(model_name: str, number_gpus: int, gpu_mem_util: float, max_model_len: int):
    if not VLLM_AVAILABLE:
        raise ImportError("vllm package not available.")
    return LLM(
        model=model_name,
        tensor_parallel_size=max(1, number_gpus),
        trust_remote_code=True,
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=max_model_len,
    )


def run_openai(client, model_name: str, messages: List[Dict[str, str]], temperature: float, top_p: float, max_new_tokens: int) -> str:
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
    )
    return response.choices[0].message.content.strip()


def run_vllm(
    llm: "LLM",
    sampling_params: "SamplingParams",
    messages: List[Dict[str, str]],
) -> str:
    prompt = messages_to_prompt(messages)
    outputs = llm.generate([prompt], sampling_params=sampling_params)
    return outputs[0].outputs[0].text.strip()


def write_record(f, record: Dict):
    f.write(json.dumps(record, ensure_ascii=False) + "\n")


def chunked(iterable, size: int):
    for i in range(0, len(iterable), size):
        yield i, iterable[i : i + size]


def main():
    parser = argparse.ArgumentParser(description="Run MedMatch prompting against multiple datasets.")
    parser.add_argument(
        "--mode",
        choices=["openai", "azure", "vllm"],
        default="openai",
        help="openai | azure (Azure OpenAI, e.g. azure-gpt-5-chat) | vllm.",
    )
    parser.add_argument("--model_name", default="gpt-4o-mini")
    parser.add_argument("--prompting_type", choices=["zero", "few", "one_shot"], default="zero",
                       help="Prompting type: 'zero' for zero-shot, 'few' for few-shot/multi-turn, 'one_shot' for single-turn one-shot")
    parser.add_argument("--num_runs", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--data_dir", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "med_match")))
    parser.add_argument("--output_dir", default="./results/med_match")
    parser.add_argument("--subset_size", type=int, default=None, help="Optional subset size for quick tests.")
    parser.add_argument("--batch_size", type=int, default=10, help="vLLM batch size (if used).")
    parser.add_argument("--number_gpus", type=int, default=2, help="Tensor parallel size for vLLM.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85, help="vLLM GPU memory utilization fraction.")
    parser.add_argument("--max_model_len", type=int, default=4096, help="vLLM maximum sequence length to reserve KV cache for.")
    args = parser.parse_args()

    # Create prompting-type specific subdirectory
    if args.prompting_type == "zero":
        prompting_subdir = "zero-shot"
    elif args.prompting_type == "few":
        prompting_subdir = "few-shot"
    elif args.prompting_type == "one_shot":
        prompting_subdir = "one-shot"
    else:
        prompting_subdir = args.prompting_type  # fallback

    output_dir = os.path.join(args.output_dir, prompting_subdir)
    os.makedirs(output_dir, exist_ok=True)
    datasets = build_datasets(args.data_dir, args.prompting_type)

    client = None
    llm = None
    sampling_params = None
    azure_model_name = None

    if args.mode == "openai":
        client = get_openai_client()
    elif args.mode == "azure":
        client, azure_model_name = get_azure_client(args.model_name)
    else:
        llm = init_vllm(
            args.model_name,
            args.number_gpus,
            args.gpu_memory_utilization,
            args.max_model_len,
        )
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_new_tokens,
        )
        tokenizer = None
        if "qwen" in args.model_name.lower() and HF_TOKENIZER_AVAILABLE:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        if "qwen" in args.model_name.lower() and tokenizer is None:
            raise RuntimeError(
                "Qwen models require transformers tokenizer to disable thinking. "
                "Install transformers or set a non-Qwen model."
            )

    for run_id in range(1, args.num_runs + 1):
        # Set fixed random seed for fully reproducible results across runs
        # random.seed(42)  # Same seed for all runs for deterministic evaluation

        output_path = os.path.join(
            output_dir, f"{sanitize_model_name(args.model_name)}_run{run_id}.jsonl"
        )
        with open(output_path, "w", encoding="utf-8") as f:
            for name, cfg in datasets.items():
                df = load_dataframe(cfg["path"], args.subset_size)

                # Prepare prompts and metadata
                prompt_rows = []
                for idx, (_, row) in enumerate(df.iterrows()):
                    messages = cfg["builder"](row[cfg["prompt_col"]])
                    
                    # Print the first data prompt
                    if run_id == 1 and idx == 0 and name == list(datasets.keys())[0]:
                        print(f"\n{'='*80}")
                        print(f"First Data Prompt - Dataset: {name}")
                        print(f"{'='*80}")
                        # print(f"Input prompt: {row[cfg['prompt_col']]}")
                        print(f"\nBuilt messages:")
                        for msg in messages:
                            content = msg.get('content', '')
                            # Print full content or truncate if too long
                            if len(content) > 500:
                                print(f"  {msg['role'].capitalize()}: {content}")
                            else:
                                print(f"  {msg['role'].capitalize()}: {content}")
                        print(f"{'='*80}\n")
                    if args.mode == "vllm":
                        if "qwen" in args.model_name.lower() and "tokenizer" in locals() and tokenizer:
                            prompt_text = tokenizer.apply_chat_template(
                                messages,
                                tokenize=False,
                                add_generation_prompt=True,
                                enable_thinking=False,
                            )
                        else:
                            prompt_text = messages_to_prompt(messages)
                        prompt_rows.append({"prompt_text": prompt_text, "row": row})
                    else:
                        prompt_rows.append({"messages": messages, "row": row})

                if args.mode in ("openai", "azure"):
                    max_workers = max(1, min(args.batch_size, len(prompt_rows)))
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = {
                            executor.submit(
                                run_openai,
                                client,
                                azure_model_name or args.model_name,
                                item["messages"],
                                args.temperature,
                                args.top_p,
                                args.max_new_tokens,
                            ): item
                            for item in prompt_rows
                        }
                        for future in tqdm(as_completed(futures), total=len(futures), desc=f"{name} run{run_id}"):
                            item = futures[future]
                            response = future.result()
                            row = item["row"]
                            record = {
                                "dataset": name,
                                "run": run_id,
                                "model": args.model_name,
                                "medication": row.get("Medication"),
                                "prompt": row[cfg["prompt_col"]],
                                "ground_truth": row.get(cfg["gt_col"]),
                                "response": response,
                            }
                            write_record(f, record)
                else:
                    # Batch vLLM requests
                    for _, batch in tqdm(
                        chunked(prompt_rows, args.batch_size),
                        total=(len(prompt_rows) + args.batch_size - 1) // args.batch_size,
                        desc=f"{name} run{run_id} (vllm batches)",
                    ):
                        batch_prompts = [b["prompt_text"] for b in batch]
                        outputs = llm.generate(batch_prompts, sampling_params=sampling_params)
                        for item, out in zip(batch, outputs):
                            response = out.outputs[0].text.strip()
                            row = item["row"]
                            record = {
                                "dataset": name,
                                "run": run_id,
                                "model": args.model_name,
                                "medication": row.get("Medication"),
                                "prompt": row[cfg["prompt_col"]],
                                "ground_truth": row.get(cfg["gt_col"]),
                                "response": response,
                            }
                            write_record(f, record)


if __name__ == "__main__":
    main()