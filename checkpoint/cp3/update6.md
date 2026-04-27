# MedMatch Logic Handoff - Update 6 (Gemma4 26B Ollama vs vLLM-MLX)

## Why This Update Exists

This update isolates a serving-stack question:

- same benchmark task: `medmatch2`
- same pipeline: two-pass normalization
- same scorer: v1 strict exact match
- same intended model family: Gemma4 26B
- different local runtime:
  - Ollama (`gemma4:26b`)
  - vLLM-MLX OpenAI-compatible API (`google/gemma-4-26B-A4B-it`)

This is not a scorer change and does not use scorer v2. It is a runtime and
model-serving comparison only.

## Branch And Code Changes

Branch:

```text
codex/gemma4-26b-vllm-mlx
```

New backend mode:

```text
local_openai
```

Purpose:

- call local OpenAI-compatible servers that are not Qwen-specific
- default to the Gemma4 26B vLLM-MLX endpoint
- keep `qwen_local` unchanged
- keep `local_openai` out of the remote prompt-style set so Gemma4 vLLM-MLX
  uses the same local normalization prompt family as Gemma4 Ollama

Default local OpenAI-compatible Gemma4 settings:

```text
LOCAL_OPENAI_BASE_URL=http://127.0.0.1:8013/v1
LOCAL_OPENAI_API_KEY=local-gemma4
LOCAL_OPENAI_MODEL_NAME=google/gemma-4-26B-A4B-it
```

Optional extra body used for this run:

```json
{"enable_thinking": false, "chat_template_kwargs": {"enable_thinking": false}}
```

The runner metadata now records:

- `temperature`
- `top_p`
- `max_new_tokens`
- `started_at`
- `completed_at`
- `elapsed_seconds`

## Runtime Setup

Gemma4 26B vLLM-MLX server:

```bash
cd /Users/tianleli/Desktop/Boulder2/Agent
HOST=127.0.0.1 PORT=8013 API_KEY=local-gemma4 \
MODEL_REPO=mlx-community/gemma-4-26b-a4b-it-4bit \
SERVED_MODEL_NAME=google/gemma-4-26B-A4B-it \
ENABLE_REASONING_PARSER=1 CONTINUOUS_BATCHING=0 \
./scripts/start_gemma4_26b_vllm_mlx_api.sh
```

API smoke result:

```text
local gemma4 api ok
```

The server still returned `reasoning_content` even with thinking disabled, but
the final `content` field contained the expected answer. The MedMatch backend
uses `content` first and only falls back to `reasoning_content` when content
is empty.

## Source Artifacts

Ollama baseline:

```text
/Users/tianleli/Desktop/Boulder2/MedMatch/results/medmatch2/model_comparison_20260425/full/gemma4_26b_rerun/normalization/gemma4:26b_run1.jsonl
/Users/tianleli/Desktop/Boulder2/MedMatch/results/medmatch2/model_comparison_20260425/full/gemma4_26b_rerun/normalization/gemma4:26b_run_metadata.json
```

vLLM-MLX smoke:

```text
results/medmatch2/runtime_comparison_20260426/smoke/gemma4_26b_vllm_mlx/normalization/google_gemma-4-26B-A4B-it_run1.jsonl
results/medmatch2/runtime_comparison_20260426/smoke/gemma4_26b_vllm_mlx/normalization/google_gemma-4-26B-A4B-it_run_metadata.json
```

vLLM-MLX full run:

```text
results/medmatch2/runtime_comparison_20260426/full/gemma4_26b_vllm_mlx/normalization/google_gemma-4-26B-A4B-it_run1.jsonl
results/medmatch2/runtime_comparison_20260426/full/gemma4_26b_vllm_mlx/normalization/google_gemma-4-26B-A4B-it_run_metadata.json
```

Summary tables:

```text
results/medmatch2/runtime_comparison_20260426/summary/
```

## Run Metadata Check

vLLM-MLX full run:

```text
rows: 101
dataset_version: medmatch2
scorer_version: v1
mode: local_openai
model_name: google/gemma-4-26B-A4B-it
temperature: 0.7
top_p: 0.95
max_new_tokens: 512
started_at: 2026-04-26T21:17:06-06:00
completed_at: 2026-04-26T21:24:12-06:00
elapsed_seconds: 426.298
```

## Overall Results

| Runtime | Exact entries | Entry accuracy | Fields | Field accuracy | Raw exact | Raw fields | Runtime |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ollama | `95/101` | `0.9406` | `867/873` | `0.9931` | `71/101` | `835/873` | `543.725s` |
| vLLM-MLX | `89/101` | `0.8812` | `859/873` | `0.9840` | `70/101` | `837/873` | `426.298s` |

Runtime delta:

- vLLM-MLX was about `117.427s` faster on this run.
- vLLM-MLX took about `78.4%` of the Ollama elapsed time.
- That is about `1.28x` the throughput by this coarse wall-clock measure.

Accuracy delta:

- vLLM-MLX was `6` exact entries lower than Ollama.
- vLLM-MLX was `8` normalized fields lower than Ollama.
- Raw field count was slightly higher for vLLM-MLX (`837` vs `835`), but the
  final normalization pass did not preserve that advantage.

## Results By Category

| Runtime | Category | Exact entries | Entry accuracy | Fields | Field accuracy |
|---|---|---:|---:|---:|---:|
| Ollama | PO solid | `39/40` | `0.9750` | `279/280` | `0.9964` |
| Ollama | PO liquid | `10/10` | `1.0000` | `100/100` | `1.0000` |
| Ollama | IV intermittent | `17/17` | `1.0000` | `136/136` | `1.0000` |
| Ollama | IV push | `17/17` | `1.0000` | `153/153` | `1.0000` |
| Ollama | IV continuous | `12/17` | `0.7059` | `199/204` | `0.9755` |
| vLLM-MLX | PO solid | `39/40` | `0.9750` | `279/280` | `0.9964` |
| vLLM-MLX | PO liquid | `9/10` | `0.9000` | `99/100` | `0.9900` |
| vLLM-MLX | IV intermittent | `17/17` | `1.0000` | `136/136` | `1.0000` |
| vLLM-MLX | IV push | `17/17` | `1.0000` | `153/153` | `1.0000` |
| vLLM-MLX | IV continuous | `7/17` | `0.4118` | `192/204` | `0.9412` |

Main finding:

- PO solid, IV intermittent, and IV push were unchanged at the entry level.
- vLLM-MLX lost one PO liquid row.
- The main accuracy drop was IV continuous: `12/17` under Ollama vs `7/17`
  under vLLM-MLX.

## Residual Error Patterns

Ollama residuals:

- PO solid Bisoprolol frequency:
  - expected `once daily`
  - actual `once daily for hypertension`
- IV continuous titration-goal surface forms:
  - Epinephrine: `map of at least 65mmhg` vs `map of 65 mmhg or greater`
  - Nicardipine: spacing difference in `120mmhg`
  - Phenylephrine: `map >= 65 mmhg` surface-form rewrite
- IV continuous Furosemide starting rate missing
- IV continuous Heparin `unit` vs `units`

vLLM-MLX residuals:

- Same PO solid Bisoprolol frequency miss.
- PO liquid Valproic acid concentration:
  - expected `50`
  - actual `250`
- Same three IV continuous titration-goal surface-form misses.
- More IV continuous starting-rate misses or unit-attached rates:
  - Propofol numerical dose missing and dose unit `10 mg/ml`
  - Milrinone starting rate missing
  - Dobutamine starting rate missing
  - Octreotide starting rate `50 mcg/hr`
  - Lidocaine starting rate missing
  - Furosemide starting rate `20 mg/hr`
  - Heparin starting rate `500 units/hr`
- Same Heparin `unit` vs `units` dose-unit miss.

## Interpretation

vLLM-MLX improved wall-clock runtime in this single run but reduced exact-entry
accuracy. The drop is concentrated in IV continuous normalization, especially
the `starting rate` field where vLLM-MLX often either omitted the scalar or
kept the scalar plus unit in the value slot.

This suggests the deployment stack can change model behavior even when the
high-level prompt, dataset, and scorer are held constant. The comparison should
therefore be described as a local runtime/serving-stack comparison, not as a
pure model-architecture comparison.

## Methodology Boundary

No scorer files were changed:

- `src/medmatch/core/scorer.py`
- `results/evaluation_match.py`
- `results/clean_results.py`

All numbers above are v1 strict exact-match results. They must not be mixed
with any future scorer v2 abbreviation-equivalency results in the same table.
If v2 is added later, produce separate v2 tables from the same JSONL artifacts.

## Tests

Unit tests run:

```bash
python -m pytest tests/test_remote_api_backend.py tests/test_qwen_local_pipeline.py tests/test_medmatch2_dataset_consistency.py tests/test_po_solid_total_dose_prompting.py
```

Result:

```text
17 passed
```
