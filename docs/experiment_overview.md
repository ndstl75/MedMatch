# MedMatch Experiment Overview

This note is a short guide to the current MedMatch experiment stack.

## Experiment Modes

- `baseline`: one extraction call that maps the order directly into MedMatch JSON
- `CoT`: a reasoning pass before extraction, used mainly for IV experiments
- `normalization`: a second LLM pass that rewrites the raw JSON into canonical MedMatch wording

The scorer stays strict in all modes: lowercase + whitespace collapse only.

## Script Map

### Remote scripts

- `scripts/run_baseline.py --backend openai|azure|google|remote`: baseline across supported categories
- `scripts/run_normalization.py --backend openai|azure|google|remote --category oral`: oral normalization for `PO Solid` and `PO liquid`
- `scripts/run_normalization.py --backend openai|azure|google|remote --category iv`: IV normalization for `IV intermittent` and `IV push`
- `scripts/run_cot.py --backend openai|azure|google|remote`: IV CoT

`remote` remains an alias for the Google-backed path so older unified-runner commands stay valid. `scripts/run_tier3.py` also remains as a compatibility alias for `scripts/run_normalization.py`.

### Local scripts

- `scripts/run_baseline.py --backend local`: unified local baseline entrypoint
- `scripts/run_cot.py --backend local`: unified local CoT entrypoint
- `scripts/run_normalization.py --backend local`: unified local normalization entrypoint
- `scripts/run_single.py --backend local`: unified single-case debug entrypoint
- `scripts/legacy/local/medmatch_test_local_appendix_exact.py`: local baseline parity script
- `scripts/legacy/local/iv_llm_normalize_conditional_local.py`: local conditional normalization parity script

## How To Read Normalization Results

Normalization scripts usually report both:

- `raw`: Pass 1 extraction only
- `normalized`: Pass 1 output after the normalization pass

Remote scoring now stays strict as well, so normalization results are judged with the same lowercase-plus-whitespace comparison used elsewhere in the unified stack.

The lift that matters is `normalized` versus `raw` within the same run.

The conditional IV normalization variant only runs Pass 2 when deterministic flags fire. If no flag fires, the raw JSON is reused as the final answer.

## Coverage Notes

- Oral normalization currently covers `PO Solid` and `PO liquid`
- IV normalization currently covers `IV intermittent` and `IV push`
- `IV continuous` is intentionally left out of normalization for now because the remaining errors are mostly clinically constrained or GT-constrained and need review rather than more normalization
