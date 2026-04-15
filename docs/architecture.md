# MedMatch Architecture

This document describes the **target** architecture for the MedMatch repository and a staged plan for migrating from the current layout to it. It is aspirational — nothing here is live yet. Use it as the reference whenever we move, rename, or consolidate code.

The three experiment families from `CLAUDE.md` and `AGENTS.md` stay first-class throughout: `baseline`, `CoT`, and `normalization` (formerly called `Tier 3`). For normalization, the distinction between `raw` extraction and `normalized` output is preserved in both code paths and result folders.

## Goals

1. One repo, one package, one mental model — a reader should be able to find baseline, CoT, or normalization code without grepping.
2. Remove the local/remote script duplication. A single experiment driver runs against either backend via a flag.
3. Keep manuscript, dataset, and generated outputs out of the source tree.
4. Make `results/` self-describing by folder, not by filename.
5. Preserve reproducibility — a migration step must never silently change a prompt, scorer, or schema.

## Non-Goals

- Rewriting the scorer or changing prompt semantics. Migration is structural only.
- Merging `IV continuous` into the normalization flow. Continuous stays out until the clinical GT audit is done.
- Publishing the dataset or manuscript files. They stay local.

## Current Layout (Snapshot)

```
MedMatch/
├── README.md, AGENTS.md, CLAUDE.md
├── iv_cot_experiment.py
├── iv_exemplar_rag_remote.py
├── iv_llm_normalize.py
├── iv_prompt_only_remote.py
├── iv_remote_common.py
├── oral_llm_normalize.py
├── iv_continuous_gt_audit.md
├── MedMatch Dataset for Experiment_ Final.xlsx
├── IV Rouge-Rx.docx
├── MedMatch Manuscript 2.17.26_ans- Kb working.docx
├── Supplemental Appendix Medmatch.docx
├── medmatch_local_gemma4/
│   ├── iv_case_prompt_experiment_local.py
│   ├── iv_clean_common_local.py
│   ├── iv_cot_experiment_local.py
│   ├── iv_exemplar_rag_local.py
│   ├── iv_llm_normalize_conditional_local.py
│   ├── iv_llm_normalize_local.py
│   ├── iv_prompt_only_local.py
│   ├── local_llm.py
│   ├── medmatch_single_test_local.py
│   ├── medmatch_test_local.py
│   ├── medmatch_test_local_appendix_exact.py
│   ├── oral_llm_normalize_local.py
│   ├── start_ollama_gemma4_workaround.sh
│   └── results/
├── checkpoint/, audit/, docs/, results/, __pycache__/
```

Pain points:

- 13 Python files at the root plus 13+ inside the old local-script area, many of which are local vs. remote twins of each other.
- Shared helpers (`iv_remote_common.py`, `iv_clean_common_local.py`) are duplicated across the local/remote split.
- Results are a flat folder of ~80+ timestamped CSV/JSON pairs with no family separation.
- Manuscript `.docx`, the dataset workbook, and `__pycache__` all live next to source code.

## Target Layout

```
MedMatch/
├── README.md
├── AGENTS.md
├── CLAUDE.md
├── pyproject.toml                      # installable package: `pip install -e .`
├── .env.example
├── .gitignore
├── data/                               # dataset workbook + any fixtures (gitignored)
│   └── MedMatch_Dataset_Final.xlsx
├── manuscript/                         # .docx, .tex, appendix — gitignored
│   ├── MedMatch_Manuscript.docx
│   ├── Supplemental_Appendix.docx
│   └── IV_Rouge-Rx.docx
├── docs/
│   ├── architecture.md                 # this file
│   ├── experiment_overview.md
│   └── iv_continuous_gt_audit.md
├── src/medmatch/
│   ├── __init__.py
│   ├── core/                           # shared, backend-agnostic
│   │   ├── dataset.py                  # workbook loader + row iterator
│   │   ├── schema.py                   # MedMatch JSON schema + field lists
│   │   ├── scorer.py                   # strict scorer (lowercase + whitespace collapse)
│   │   ├── io.py                       # result writers (CSV + JSON, consistent paths)
│   │   └── cleaning.py                 # replaces iv_*_common helpers
│   ├── llm/
│   │   ├── base.py                     # LLMBackend interface: .generate(prompt, **kw)
│   │   ├── remote_gemma.py             # google-genai, gemma-3-27b-it
│   │   └── local_ollama.py             # replaces the old local_llm helper
│   ├── prompts/
│   │   ├── baseline/                   # one prompt per category
│   │   ├── cot/                        # IV CoT reasoning prompts
│   │   └── normalization/              # oral + IV normalization prompts
│   └── experiments/
│       ├── baseline.py                 # single-pass extraction, all categories
│       ├── cot.py                      # IV intermittent + IV push CoT
│       └── tier3_normalize.py          # oral + IV normalization, raw + normalized
├── scripts/                            # thin CLI drivers — user-facing entry points
│   ├── run_baseline.py                 #   --backend {local,remote} --category ...
│   ├── run_cot.py
│   ├── run_normalization.py
│   └── run_tier3.py                    # compatibility alias
├── results/                            # gitignored
│   ├── baseline/
│   │   └── 2026-04-13_iv_push_local/
│   │       ├── predictions.csv
│   │       └── run.json                # metadata: backend, model, prompt hash, seed
│   ├── cot/
│   └── normalization/
│       └── 2026-04-13_iv_push_local/
│           ├── raw.csv                 # raw extraction pass
│           ├── raw.json
│           ├── normalized.csv          # second-pass LLM normalization
│           └── normalized.json
├── audit/                              # gitignored — GT audits, reviewer spreadsheets
├── checkpoint/                         # gitignored — in-progress model state
└── tests/
    ├── test_scorer.py
    ├── test_schema.py
    └── test_prompts.py
```

## Key Consolidations

### 1. Backend abstraction collapses the local/remote split

```
src/medmatch/llm/base.py
    class LLMBackend:
        def generate(self, prompt: str, **kw) -> str: ...

src/medmatch/llm/remote_gemma.py   # google-genai client
src/medmatch/llm/local_ollama.py   # ollama client, gemma-3 local
```

Every experiment driver takes a backend instance, so `iv_prompt_only_remote.py` and `iv_prompt_only_local.py` collapse into one code path selected by `--backend`. The Ollama bootstrap shell script (`start_ollama_gemma4_workaround.sh`) moves to `scripts/`.

### 2. Three experiment families, one module each

| Family | Module | Categories |
| --- | --- | --- |
| baseline | `experiments/baseline.py` | PO Solid, PO Liquid, IV Intermittent, IV Push, IV Continuous |
| CoT | `experiments/cot.py` | IV Intermittent, IV Push |
| normalization | `experiments/tier3_normalize.py` | PO Solid, PO Liquid, IV Intermittent, IV Push |

IV Continuous stays out of the normalization module by design, matching the current project guidance.

### 3. Shared helpers live in `core/`, not in per-script `*_common.py`

`iv_remote_common.py` + `iv_clean_common_local.py` merge into `core/cleaning.py`. Both backends import the same helpers, so prompt-adjacent logic cannot drift silently.

### 4. Results are self-describing by folder

Current: `results/IV_push_17_iv_llm_norm_20260413_181603.csv`
Target: `results/normalization/2026-04-13_iv_push_local/normalized.csv`

Each run folder contains a `run.json` with:

- backend (`local` or `remote`)
- model name + version
- prompt template hash
- category and split
- git commit hash
- timestamp

For normalization runs, `raw.*` and `normalized.*` are always both present. This is the CLAUDE.md rule encoded in the directory layout.

### 5. Scripts are thin drivers

Example:

```bash
python scripts/run_normalization.py --backend local --category iv_push
python scripts/run_baseline.py --backend remote --category po_solid
```

The actual logic lives in `src/medmatch/experiments/*`. Scripts do arg parsing, backend instantiation, and call into the experiment module.

## Migration Plan

Each step is its own small, scoped commit, matching the pre-push checklist in `CLAUDE.md`.

### Step 1 — Hygiene only, no code movement

- Tighten `.gitignore` (`__pycache__/**`, `.DS_Store` anywhere, `audit/`, `~$*` Office lockfiles, `input.txt`, `results/` subtrees).
- Remove any already-tracked `__pycache__` or `.DS_Store` from the index.

### Step 2 — Move non-code into dedicated folders

- `manuscript/` ← all `.docx`, `.tex`, appendix files (stays gitignored).
- `data/` ← the dataset workbook (stays gitignored).
- `audit/` stays but is explicitly gitignored.
- No code changes.

### Step 3 — Introduce `src/medmatch/` as an additive layer

- Create `src/medmatch/core/`, `src/medmatch/llm/`, `src/medmatch/prompts/`, `src/medmatch/experiments/`.
- Extract shared helpers (scorer, schema, dataset loader, IO, IV cleaning helpers, Ollama client) into `core/` and `llm/`.
- Old root scripts keep working by importing from `medmatch.*` instead of re-implementing. No behavior change.
- Add `pyproject.toml` and `pip install -e .` so imports work.

### Step 4 — Add unified CLI drivers under `scripts/`

- `scripts/run_baseline.py`, `scripts/run_cot.py`, `scripts/run_normalization.py`, `scripts/run_tier3.py`.
- Each driver accepts `--backend` and `--category`. Under the hood they call into `src/medmatch/experiments/*`.
- Reproduce one existing run per family with the new driver; diff the CSV against the old output. They must match byte-for-byte before moving on.

Status in this refactor branch:
- `baseline.py`, `cot.py`, and `tier3_normalize.py` now contain the active unified runners.
- `scripts/legacy/` and `scripts/legacy/local/` are retained only as migration references and parity fallbacks.
- Local parity notes now live in:
  - `docs/refactor_parity.md` for the fixed `iv_push` slice
  - `docs/refactor_parity_iv_intermittent.md` for the fixed `iv_intermittent` slice
- Current parity status:
  - `baseline`, `CoT`, and `normalization` show zero deltas on both local slices

### Step 5 — Delete the duplicated local/remote scripts

- Once every family has a verified unified driver, remove:
  - `scripts/legacy/iv_prompt_only_remote.py` + `scripts/legacy/local/iv_prompt_only_local.py`
  - `scripts/legacy/iv_cot_experiment.py` + `scripts/legacy/local/iv_cot_experiment_local.py`
  - `scripts/legacy/iv_llm_normalize.py` + `scripts/legacy/local/iv_llm_normalize_local.py` + `scripts/legacy/local/iv_llm_normalize_conditional_local.py`
  - `scripts/legacy/oral_llm_normalize.py` + `scripts/legacy/local/oral_llm_normalize_local.py`
- Keep `scripts/legacy/local/medmatch_test_local_appendix_exact.py` until baseline parity is verified, then remove.

### Step 6 — Results folder rework

- Write a small converter that reads an existing timestamped filename and moves it into the new family/date/backend folder structure. Run once over `results/` and the old local results tree.
- Update `io.py` so new runs only ever write the new layout.

### Step 7 — Docs and entry-point updates

- `README.md`: replace the current script map with the three `scripts/run_*.py` entry points.
- `AGENTS.md` and `CLAUDE.md`: update "Important Active Scripts" to the new paths.
- `docs/experiment_overview.md`: keep as the conceptual overview, point to `architecture.md` for layout.

## Reproducibility Rules During Migration

- Never change a prompt string and a file location in the same commit.
- Every refactor commit must be able to reproduce at least one prior run's CSV exactly. If it cannot, the commit is not ready.
- Normalization raw vs normalized separation is mandatory at every step. If a refactor collapses them, revert it.
- IV Continuous is not swept into normalization by any step of the migration.

## Open Questions

- Do we want a unified experiment config (YAML / TOML) per run, or keep CLI flags only? YAML makes sweeps easier but adds a dependency.
- Where should `checkpoint/` live long-term — inside `results/` or as a sibling? Leaving as sibling for now.
