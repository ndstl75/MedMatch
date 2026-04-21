# MedMatch

**Structured medication order formatting for LLMs and downstream evaluation.**

Python 3.10+ · Prompting backends: local Ollama, OpenAI-compatible APIs, Azure OpenAI, Google GenAI, vLLM (optional)

MedMatch turns free-text medication instructions into a fixed JSON slot schema (drug name, dose, units, route, frequency, and route-specific fields for oral vs intravenous orders). This repository contains prompt builders, dataset CSVs, runners that log model outputs as JSONL, and scripts to score routes, entity-level agreement, and survey-based appropriateness.

---

## Core idea

1. **Slot-filling JSON** — Each formulation and administration class (e.g. oral solid, IV intermittent) has a defined set of keys; prompts require the model to output a single JSON object with those keys.
2. **Multi-setting evaluation** — Same codebase supports zero-shot, few-shot, and one-shot prompting, plus auxiliary tasks such as **route-only** prediction from de-routed text.
3. **Traceable runs** — The canonical runner writes JSONL files under `results/` so you can reproduce and aggregate scores offline.

---

## Repository layout

```
MedMatch/
├── src/                         # Prompt builders + shared backend helpers
│   ├── prompt_medmatch.py       # MedMatch formatting prompts (PO / IV)
│   ├── prompt_rougerx.py        # RougeRx component-extraction prompts
│   └── medmatch/                # Shared scorer / backend / dataset helpers
├── scripts/                     # Entry points (run from repo root)
│   ├── probing_medmatch.py      # Canonical MedMatch runner: baseline, CoT, normalization
│   ├── probing_medmatch_route_selection_test.py
│   ├── survey2gpt5.py           # Survey 2 appropriateness + tables
│   ├── rougerx.py               # RougeRx survey analysis
│   ├── run_cot.py               # Compatibility wrapper for CoT
│   ├── run_normalization.py     # Compatibility wrapper for normalization
│   ├── run_tier3.py             # Compatibility alias for normalization
│   └── run_single.py            # Single-case debugging helper
├── data/
│   ├── med_match/               # MedMatch CSV benchmarks
│   ├── survey1/                 # RougeRx CSV
│   └── survey2/                 # Second-survey CSVs
├── datasets/                    # Workbook inputs retained for legacy/reference scripts
├── results/                     # JSONL outputs + derived tables (git may omit large files)
├── requirements.txt
└── README.md
```

The primary workflow is now centralized in `scripts/probing_medmatch.py`. The `run_cot.py` / `run_normalization.py` wrappers are retained for compatibility, while `run_single.py` is a debugger for one-off inspection.

---

## Data layout

- **`data/med_match/`** — Per-task CSVs (oral solid/liquid, IV push/intermittent/continuous, with or without route columns). `scripts/probing_medmatch.py` uses these CSVs as the canonical MedMatch benchmark source.
- **`data/survey2/`** — CSVs for the “computer-generated survey” appropriateness task consumed by `scripts/survey2gpt5.py`.
- **`data/survey1/rougerx.csv`** — RougeRx respondent data for `scripts/rougerx.py`.
- **`datasets/MedMatch Dataset for Experiment_ Final.xlsx`** — Workbook retained for legacy/reference scripts.

---

## Installation

```bash
git clone https://github.com/ndstl75/MedMatch.git
cd MedMatch
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Set credentials in the environment or a `.env` file (never commit secrets):

- **OpenAI-compatible API:** `OPENAI_API_KEY` and optionally `OPENAI_BASE_URL`, `OPENAI_MODEL`, `OPENAI_EXTRA_BODY_JSON`
- **Azure OpenAI:** `AZURE_OPENAI_ENDPOINT` (your resource URL, e.g. `https://<resource-name>.openai.azure.com`) plus Azure identity / CLI login as required by `azure-identity`
- **Google-backed remote path:** `GOOGLE_API_KEY`
- **Local Qwen 3.6 via vLLM-MLX OpenAI API:** `LOCAL_OPENAI_BASE_URL`, `LOCAL_OPENAI_API_KEY`, `LOCAL_OPENAI_MODEL_NAME`, `LOCAL_OPENAI_EXTRA_BODY` (falls back to `OPENAI_*`)

For vLLM, install the optional `vllm` / `transformers` stack in the same environment.

Download **NLTK** tokenizers once if you use RougeRx:

```bash
python -c "import nltk; nltk.download('punkt')"
```

---

## Quick start

Run all commands from the **repository root** so default paths resolve.

### MedMatch formatting (baseline / CoT / normalization)

```bash
python scripts/probing_medmatch.py \
  --mode openai \
  --model_name gpt-4o-mini \
  --prompting_type zero \
  --num_runs 3 \
  --temperature 0.7 \
  --batch_size 10
```

Outputs default to `results/med_match/` (override with `--output_dir`). Use `--data_dir` to point at another MedMatch CSV folder.

CoT and normalization use the same runner:

```bash
python scripts/probing_medmatch.py \
  --mode openai \
  --model_name gpt-4o-mini \
  --prompting_type cot \
  --num_runs 1 \
  --subset_size 1

python scripts/probing_medmatch.py \
  --mode local \
  --model_name gemma4:e4b \
  --prompting_type normalization \
  --num_runs 1 \
  --subset_size 1
```

Local Qwen 3.6 over your OpenAI-compatible vLLM-MLX endpoint:

```bash
export LOCAL_OPENAI_BASE_URL="http://127.0.0.1:8011/v1"
export LOCAL_OPENAI_API_KEY="local-qwen"
export LOCAL_OPENAI_MODEL_NAME="Qwen/Qwen3.6-35B-A3B"

python scripts/probing_medmatch.py \
  --mode qwen_local \
  --prompting_type zero \
  --num_runs 1 \
  --subset_size 1
```

`qwen_local` automatically sends `chat_template_kwargs.enable_thinking=false` unless you override it with `LOCAL_OPENAI_EXTRA_BODY`.

Category support:

- `zero`, `few`, `one_shot`: all five MedMatch CSV tasks
- `cot`: IV intermittent, IV push, IV continuous
- `normalization`: PO solid, PO liquid, IV intermittent, IV push, IV continuous

### Route selection only

```bash
python scripts/probing_medmatch_route_selection_test.py \
  --mode openai \
  --model_name gpt-4o-mini \
  --num_runs 3 \
  --output_dir results/route
```

### Survey 2 (appropriateness / Percentage Appropriate table)

```bash
python scripts/survey2gpt5.py --mode openai --model_name gpt-4o-mini
```

### RougeRx (quick import test)

```bash
PYTHONPATH=. python -c "from scripts.rougerx import quick_test; quick_test()"
```

Or run the full script pipeline:

```bash
python scripts/rougerx.py
```

### Compatibility wrappers and debugger

```bash
python3 scripts/run_cot.py --backend local --category iv
python3 scripts/run_normalization.py --backend local --category iv_push
python3 scripts/run_single.py --backend local --category iv_push --prompt "..."
```

`scripts/run_cot.py` and `scripts/run_normalization.py` now delegate to `scripts/probing_medmatch.py`. `scripts/run_tier3.py` is kept as a compatibility alias for `scripts/run_normalization.py`. `scripts/run_single.py` is a debugger, not the main batch runner.

---

## Evaluation utilities

Python modules under `results/` (e.g. `evaluation_match.py`, `evaluation_route.py`, `convert_table.py`) consume JSONL outputs and emit CSV / JSON summaries. Paths inside those scripts may still be machine-specific in places; adjust file locations to match your clone.

---

## Citation

If you use this repository in research, please cite the relevant paper(s) for your project and, if applicable, add:

```bibtex
@misc{medmatch2026codebase,
  title        = {MedMatch: Structured Medication Order Formatting and Evaluation},
  howpublished = {\url{https://github.com/AIChemist-Lab/MedMatch}},
  year         = {2026}
}
```

---

## License

Add a `LICENSE` file to the repository root if you intend to distribute the code; until then, all rights are reserved unless you specify otherwise.
