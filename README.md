# MedMatch

**Structured medication order formatting for LLMs and downstream evaluation.**

Python 3.10+ · Prompting backends: OpenAI-compatible APIs, Azure OpenAI, vLLM (optional)

MedMatch turns free-text medication instructions into a fixed JSON slot schema (drug name, dose, units, route, frequency, and route-specific fields for oral vs intravenous orders). This repository contains prompt builders, dataset CSVs, runners that log model outputs as JSONL, and scripts to score routes, entity-level agreement, and survey-based appropriateness.

---

## Core idea

1. **Slot-filling JSON** — Each formulation and administration class (e.g. oral solid, IV intermittent) has a defined set of keys; prompts require the model to output a single JSON object with those keys.
2. **Multi-setting evaluation** — Same codebase supports zero-shot, few-shot, and one-shot prompting, plus auxiliary tasks such as **route-only** prediction from de-routed text.
3. **Traceable runs** — Runners write JSONL files under `results/` so you can reproduce and aggregate scores offline.

---

## Repository layout

```
MedMatch/
├── src/                         # Prompt builders
│   ├── prompt_medmatch.py       # MedMatch formatting prompts (PO / IV)
│   ├── prompt_rougerx.py        # RougeRx component-extraction prompts
│   └── medmatch/                # Shared helpers used by merged-in local/unified runners
├── scripts/                     # Entry points (run from repo root)
│   ├── probing_medmatch.py      # Full MedMatch formatting experiments
│   ├── probing_medmatch_route_selection_test.py
│   ├── survey2gpt5.py           # Survey 2 appropriateness + tables
│   ├── rougerx.py               # RougeRx survey analysis
│   ├── run_baseline.py          # Added baseline runner
│   ├── run_cot.py               # Added CoT runner
│   ├── run_normalization.py     # Added normalization runner
│   └── run_single.py            # Added single-case debugging runner
├── data/
│   ├── med_match/               # MedMatch CSV benchmarks
│   ├── survey1/                 # RougeRx CSV
│   └── survey2/                 # Second-survey CSVs
├── datasets/                    # Workbook inputs used by added IV / normalization flows
├── results/                     # JSONL outputs + derived tables (git may omit large files)
├── requirements.txt
└── README.md
```

The original lab scripts remain the primary public workflow. The added `run_*.py` scripts are supplemental entry points for local debugging and merged-in follow-up experiments; they are meant to fit inside the same repo layout rather than replace it.

---

## Data layout

- **`data/med_match/`** — Per-task CSVs (oral solid/liquid, IV push/intermittent/continuous, with or without route columns). Baseline runners default to this directory.
- **`data/survey2/`** — CSVs for the “computer-generated survey” appropriateness task consumed by `scripts/survey2gpt5.py`.
- **`data/survey1/rougerx.csv`** — RougeRx respondent data for `scripts/rougerx.py`.
- **`datasets/MedMatch Dataset for Experiment_ Final.xlsx`** — Workbook used by the added CoT / normalization follow-up flows.

---

## Installation

```bash
git clone https://github.com/AIChemist-Lab/MedMatch.git
cd MedMatch
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Set credentials in the environment or a `.env` file (never commit secrets):

- **OpenAI-compatible API:** `OPENAI_API_KEY`
- **Azure OpenAI:** `AZURE_OPENAI_ENDPOINT` (your resource URL, e.g. `https://<resource-name>.openai.azure.com`) plus Azure identity / CLI login as required by `azure-identity`
- **Google-backed remote path:** `GOOGLE_API_KEY` when using the added `run_*.py` remote helpers

For vLLM, install the optional `vllm` / `transformers` stack in the same environment.

Download **NLTK** tokenizers once if you use RougeRx:

```bash
python -c "import nltk; nltk.download('punkt')"
```

---

## Quick start

Run all commands from the **repository root** so default paths resolve.

### MedMatch formatting (zero-shot example)

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

### Added local / follow-up runners

```bash
python3 scripts/run_baseline.py --backend local --category iv_push
python3 scripts/run_cot.py --backend local --category iv
python3 scripts/run_normalization.py --backend local --category iv_push
python3 scripts/run_single.py --backend local --category iv_push --prompt "..."
```

`scripts/run_tier3.py` is kept as a compatibility alias for `scripts/run_normalization.py`.

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
