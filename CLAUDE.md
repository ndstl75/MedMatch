# CLAUDE.md

## MedMatch Quick Guide

Use this file as a short orientation note for future coding work in this repo.

## What This Repo Is Now

- `main` is the active integrated branch.
- The current unified experiment surface is:
  - `baseline`
  - `CoT`
  - `normalization`
  - `run_single`
- The unified `exemplar-RAG` path is no longer part of the active interface.
- `scripts/run_tier3.py` is retained only as a compatibility alias to `scripts/run_normalization.py`.

## Primary Commands

Run from the repo root.

```bash
python3 scripts/run_baseline.py --backend local --category iv_push
python3 scripts/run_cot.py --backend local --category iv
python3 scripts/run_normalization.py --backend local --category iv_push
python3 scripts/run_single.py --backend local --category iv_push --prompt "..."
```

Compatibility alias:

```bash
python3 scripts/run_tier3.py --backend local --category iv_push
```

## Lab Scripts To Keep Working

- `scripts/probing_medmatch.py`
- `scripts/probing_medmatch_route_selection_test.py`
- `scripts/survey2gpt5.py`
- `scripts/rougerx.py`

These are not the preferred unified interface, but they are still part of the repository contract.

## Implementation Rules

- Baseline changes are high-risk:
  - keep baseline aligned with the paper/lab path
  - do not casually change baseline prompts, schema, or scoring
- Use `normalization` as the public-facing name for the two-call rewrite flow.
- Keep strict comparison semantics unless the task explicitly requires scorer changes.
- Avoid reintroducing unified `RAG` unless the user explicitly asks for it.

## File Hygiene

- Do not commit:
  - `.env`
  - `results/`
  - `checkpoint/`
  - `audit/`
  - `__pycache__/`
  - local notes or PI-only drafts
- Prefer source code, stable docs, and tests in commits.

## Local Backend Notes

- Ollama defaults:
  - `OLLAMA_BASE_URL=http://localhost:11434`
  - `OLLAMA_MODEL=gemma4:e4b`
- Useful env knobs:
  - `MEDMATCH_NUM_RUNS`
  - `MEDMATCH_MAX_ENTRIES`
  - `OLLAMA_TIMEOUT_SECONDS`

## Current Mental Model

- `baseline`: one-call structured extraction
- `CoT`: reasoning before extraction, mostly IV-focused
- `normalization`: extract first, then rewrite into canonical wording
- `run_single`: ad hoc debugging path

When making changes, prefer preserving compatibility and making the current project shape easier to understand.
