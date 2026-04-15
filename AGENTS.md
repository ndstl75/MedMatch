# MedMatch Agent Notes

This repository contains lab scripts, unified experiment runners, datasets, and manuscript-adjacent files. Keep changes small, explicit, and easy to review.

## Current Project Shape

- Main active user-facing flows:
  - `baseline`
  - `CoT`
  - `normalization`
  - `run_single`
- The unified `exemplar-RAG` path has been removed from the active stack.
- `scripts/run_tier3.py` remains only as a compatibility alias for `scripts/run_normalization.py`.
- Original lab scripts are still part of the repo and should remain usable:
  - `scripts/probing_medmatch.py`
  - `scripts/probing_medmatch_route_selection_test.py`
  - `scripts/survey2gpt5.py`
  - `scripts/rougerx.py`

## Active Entrypoints

- Prefer these unified runners for new work:
  - `scripts/run_baseline.py`
  - `scripts/run_cot.py`
  - `scripts/run_normalization.py`
  - `scripts/run_single.py`
- Keep `scripts/run_tier3.py` working as an alias unless explicitly asked to remove compatibility.
- Treat `scripts/legacy/` and `scripts/legacy/local/` as reference/parity code, not the main interface.

## Behavior Guardrails

- Preserve paper-faithful baseline behavior:
  - baseline should stay CSV-backed where the paper/lab path expects CSV-backed evaluation
  - baseline prompt/schema/scoring changes are high-risk and should be called out explicitly
- Preserve strict scoring semantics unless the task is specifically about scorer changes.
- Do not silently reintroduce `RAG` into the unified public workflow.
- Prefer the term `normalization` over `Tier 3` in user-facing docs and commands.

## Secrets And Private Files

- Never hardcode API keys or credentials.
- Read secrets from environment variables or local `.env`.
- Keep `.env`, local notes, caches, checkpoints, results, and machine-specific artifacts out of commits.
- Do not commit PI-facing drafts or private progress notes.

## Repository Hygiene

- Before pushing, verify `git status` only shows intentional source/doc changes.
- Do not commit generated files from `results/`, `checkpoint/`, `audit/`, or `__pycache__/`.
- Keep commits scoped to one task.
- If a change affects reproducibility, prompt logic, scoring, or public-facing workflow, mention it clearly.

## Local Testing Notes

- Local Ollama backend reads:
  - `OLLAMA_BASE_URL`
  - `OLLAMA_MODEL`
  - `OLLAMA_TIMEOUT_SECONDS`
- Current local smoke tests have been exercised with `gemma4:e4b` for:
  - `baseline`
  - `normalization`

## Working Style

- Prefer minimal diffs over broad cleanup.
- If you find an old branch of logic that is no longer part of the active stack, verify whether it is legacy-only before editing it.
- When in doubt, preserve lab functionality and add compatibility rather than replacing behavior outright.
