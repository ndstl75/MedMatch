# MedMatch Experiment Overview

This note is a short guide to the current MedMatch experiment stack.

## Experiment Modes

- `baseline`: one extraction call that maps the order directly into MedMatch JSON
- `CoT`: a reasoning pass before extraction, used mainly for IV experiments
- `Tier 3` / `LLM normalization`: a second LLM pass that rewrites the raw JSON into canonical MedMatch wording

The scorer stays strict in all modes: lowercase + whitespace collapse only.

## Script Map

### Remote scripts

- `scripts/run_baseline.py --backend openai|azure|google|remote`: baseline across supported categories
- `scripts/run_tier3.py --backend openai|azure|google|remote --category oral`: oral Tier 3 for `PO Solid` and `PO liquid`
- `scripts/run_tier3.py --backend openai|azure|google|remote --category iv`: IV Tier 3 for `IV intermittent` and `IV push`
- `scripts/run_cot.py --backend openai|azure|google|remote`: IV CoT
- `scripts/run_exemplar_rag.py --backend openai|azure|google|remote`: IV exemplar-RAG

`remote` remains an alias for the Google-backed path so older unified-runner commands stay valid.

### Local scripts

- `scripts/run_baseline.py --backend local`: unified local baseline entrypoint
- `scripts/run_cot.py --backend local`: unified local CoT entrypoint
- `scripts/run_tier3.py --backend local`: unified local Tier 3 entrypoint
- `scripts/run_single.py --backend local`: unified single-case debug entrypoint
- `scripts/legacy/local/medmatch_test_local_appendix_exact.py`: local baseline parity script
- `scripts/legacy/local/iv_llm_normalize_conditional_local.py`: local conditional Tier 3 parity script

## How To Read Tier 3 Results

Tier 3 scripts usually report both:

- `raw`: Pass 1 extraction only
- `normalized`: Pass 1 output after the Tier 3 repair pass

Remote scoring still uses `normalize_relaxed`, which is intentionally preserved from the pre-refactor remote scripts so scorer semantics do not drift during the architecture migration.

The lift that matters is `normalized` versus `raw` within the same run.

The conditional IV Tier 3 variant only runs Pass 2 when deterministic flags fire. If no flag fires, the raw JSON is reused as the final answer.

## Coverage Notes

- Oral Tier 3 currently covers `PO Solid` and `PO liquid`
- IV Tier 3 currently covers `IV intermittent` and `IV push`
- `IV continuous` is intentionally left out of Tier 3 for now because the remaining errors are mostly clinically constrained or GT-constrained and need review rather than more normalization
