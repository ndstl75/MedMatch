# MedMatch

MedMatch is a medication-order structuring and evaluation project. This repository now carries both the original lab prompt-and-analysis workflow and a newer package-backed experiment stack for baseline, CoT, normalization, and single-case debugging.

## Repository Layout

- `src/`: original prompt builders used by the lab scripts
- `src/medmatch/`: shared package code for unified runners, scoring, dataset loading, and LLM backends
- `scripts/probing_medmatch.py`: original MedMatch formatting experiments
- `scripts/probing_medmatch_route_selection_test.py`: original route-selection experiments
- `scripts/survey2gpt5.py` and `scripts/rougerx.py`: survey analysis entry points
- `scripts/run_baseline.py`, `scripts/run_cot.py`, `scripts/run_normalization.py`, `scripts/run_tier3.py`, `scripts/run_single.py`: unified experiment entry points
- `scripts/legacy/` and `scripts/legacy/local/`: migration parity/reference scripts
- `scripts/local/`: local helper scripts such as the Ollama startup workaround
- `data/` and `datasets/`: lab datasets and workbook inputs already tracked in the repository
- `docs/`: experiment notes, parity reports, and local workflow guidance

## Experiment Modes

- `baseline`: one extraction call that maps an order directly into MedMatch JSON
- `CoT`: a reasoning pass before extraction, used mainly for IV experiments
- `normalization`: a second LLM pass that rewrites raw JSON into stricter canonical wording while keeping the scorer behavior stable

For a short script map, see [`docs/experiment_overview.md`](docs/experiment_overview.md). For remote parity notes, see [`docs/refactor_remote_parity.md`](docs/refactor_remote_parity.md).

## Setup

The unified runners are packaged through `pyproject.toml`:

```bash
cd "$(git rev-parse --show-toplevel)"
pip install -e .
```

Set credentials in the environment or a local `.env` file:

- `GOOGLE_API_KEY` for `--backend remote` or `--backend google`
- `OPENAI_API_KEY` for `--backend openai`
- `AZURE_OPENAI_ENDPOINT` plus Azure login/identity for `--backend azure`
- `MEDMATCH_NUM_RUNS`, `MEDMATCH_RETRY_DELAY`, `MEDMATCH_SLEEP_SECONDS`, `MEDMATCH_SHEETS`, and `MEDMATCH_MAX_ENTRIES` as optional runtime controls

This repository ignores `.env`; never commit credentials.

## Quick Start

Run commands from the repository root:

```bash
cd "$(git rev-parse --show-toplevel)"
```

Unified baseline:

```bash
MEDMATCH_NUM_RUNS=1 python3 scripts/run_baseline.py --backend openai --category all
```

Unified CoT:

```bash
python3 scripts/run_cot.py --backend azure --category iv
```

Unified normalization:

```bash
python3 scripts/run_normalization.py --backend remote --category iv
```

`scripts/run_tier3.py` remains as a compatibility alias for the normalization runner, and `remote` remains a compatibility alias for the Google-backed remote path, so older commands still work.

Single ad-hoc local case:

```bash
python3 scripts/run_single.py --backend local --category iv_push --prompt "Famotidine 20 mg, 2 mL of a 20 mg/2 mL vial, was administered twice daily via intravenous push."
```

Original lab prompt experiments remain available:

```bash
python3 scripts/probing_medmatch.py --mode openai --model_name gpt-4o-mini --prompting_type zero --num_runs 3
python3 scripts/probing_medmatch_route_selection_test.py --mode openai --model_name gpt-4o-mini --num_runs 3
```

## Local Workflow

For the local Ollama setup and the preserved local parity scripts, see [`docs/local_workflow.md`](docs/local_workflow.md).

## Repository Hygiene

Generated outputs, caches, checkpoints, local review artifacts, private notes, and local `.env` files should stay out of Git. Before pushing, verify that staged changes contain only intentional source code and stable docs.
