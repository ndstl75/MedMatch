"""
Route selection testing for MedMatch medication data.

Loads MedMatch CSVs with a "Medication JSON without route" column and tests route
selection using the route selection prompt.

Examples (from repository root):

    python scripts/probing_medmatch_route_selection_test.py --mode openai \\
        --model_name gpt-4o-mini --num_runs 3 --output_dir results/route

    CUDA_VISIBLE_DEVICES=0,1 python scripts/probing_medmatch_route_selection_test.py \\
        --mode vllm --model_name meta-llama/Llama-3.3-70B-Instruct --batch_size 30

Azure OpenAI: set AZURE_OPENAI_ENDPOINT and run with --mode azure.
"""

import argparse
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _REPO_ROOT)

import pandas as pd
from tqdm import tqdm

from src.prompt_medmatch import build_route_selection_messages

# Optional providers
try:
    from openai import OpenAI, AzureOpenAI
    OPENAI_AVAILABLE = True
    AZURE_AVAILABLE = True
except ImportError:
    try:
        from openai import OpenAI
        OPENAI_AVAILABLE = True
        AZURE_AVAILABLE = False
    except ImportError:
        OPENAI_AVAILABLE = False
        AZURE_AVAILABLE = False

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


def build_route_datasets(base_dir: str) -> Dict[str, Dict]:
    """Configure datasets for route selection task."""
    return {
        "po_solid": {
            "path": os.path.join(base_dir, "med_match - po_solid_no_route.csv"),
            "prompt_col": "Medication JSON without route",
            "gt_col": "route",
            "builder": build_route_selection_messages,
        },
        "po_liquid": {
            "path": os.path.join(base_dir, "med_match - po_liquid_no_route.csv"),
            "prompt_col": "Medication JSON without route",
            "gt_col": "route",
            "builder": build_route_selection_messages,
        },
        "iv_intermit": {
            "path": os.path.join(base_dir, "med_match - iv_i_no_route.csv"),
            "prompt_col": "Medication JSON without route",
            "gt_col": "Route",
            "builder": build_route_selection_messages,
        },
        "iv_push": {
            "path": os.path.join(base_dir, "med_match - iv_i_p_route.csv"),
            "prompt_col": "Medication JSON without route",
            "gt_col": "Route",
            "builder": build_route_selection_messages,
        },
        "iv_continuous": {
            "path": os.path.join(base_dir, "med_match - iv_c_no_route.csv"),
            "prompt_col": "Medication JSON without route",
            "gt_col": "Route",
            "builder": build_route_selection_messages,
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


def get_azure_client(model_name: str):
    """
    Create Azure OpenAI client and deployment name.
    model_name: e.g. 'azure-gpt-5-chat' or 'gpt-5-chat' (with --mode azure).
    Returns (client, deployment_name).
    """
    if not AZURE_AVAILABLE or not AZURE_IDENTITY_AVAILABLE:
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
            "(e.g. https://<resource-name>.openai.azure.com)"
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


def run_vllm(llm: "LLM", sampling_params: "SamplingParams", messages: List[Dict[str, str]]) -> str:
    prompt = messages_to_prompt(messages)
    outputs = llm.generate([prompt], sampling_params=sampling_params)
    return outputs[0].outputs[0].text.strip()


def write_record(f, record: Dict):
    f.write(json.dumps(record, ensure_ascii=False) + "\n")


def chunked(iterable, size: int):
    for i in range(0, len(iterable), size):
        yield i, iterable[i : i + size]


def main():
    parser = argparse.ArgumentParser(description="Test route selection on MedMatch datasets.")
    parser.add_argument("--mode", choices=["openai", "azure", "vllm"], default="openai")
    parser.add_argument("--model_name", default="gpt-4o-mini")
    parser.add_argument("--num_runs", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=50)  # Shorter for route selection
    parser.add_argument("--data_dir", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "med_match")))
    parser.add_argument(
        "--output_dir",
        default=os.path.join(_REPO_ROOT, "results", "route"),
    )
    parser.add_argument("--subset_size", type=int, default=None, help="Optional subset size for quick tests.")
    parser.add_argument("--batch_size", type=int, default=10, help="vLLM batch size (if used).")
    parser.add_argument("--number_gpus", type=int, default=2, help="Tensor parallel size for vLLM.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85, help="vLLM GPU memory utilization fraction.")
    parser.add_argument("--max_model_len", type=int, default=4096, help="vLLM maximum sequence length to reserve KV cache for.")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    datasets = build_route_datasets(args.data_dir)

    client = None
    llm = None
    sampling_params = None
    model_param = args.model_name

    if args.mode == "openai":
        client = get_openai_client()
    elif args.mode == "azure":
        client, model_param = get_azure_client(args.model_name)
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
        random.seed(42)

        output_path = os.path.join(
            args.output_dir, f"{sanitize_model_name(args.model_name)}_run{run_id}.jsonl"
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
                        print(f"\nBuilt messages:")
                        for msg in messages:
                            content = msg.get('content', '')
                            # Print full content or truncate if too long
                            if len(content) > 500:
                                print(f"  {msg['role'].capitalize()}: {content[:500]}...")
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
                                model_param,
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