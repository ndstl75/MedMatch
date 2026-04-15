# Refactor Parity Note

- backend: `local`
- category: `iv_push`
- python: `/opt/anaconda3/envs/medmatch/bin/python3`
- ollama model: `medgemma:27b`

## Results

### baseline
- acceptance: `pass`
- unified json: `/Users/tianleli/Desktop/Boulder2/MedMatch-refactor/results/IV_push_17_local_baseline_20260413_233254.json`
- legacy json: `/Users/tianleli/Desktop/Boulder2/MedMatch-refactor/scripts/legacy/local/results/IV_push_17_dataset_aligned_appendix_style_20260413_233307.json`
- overall_delta: `0`
- field_delta: `0`
- json_keys_match: `True`
- csv_header_match: `True`

### cot
- acceptance: `pass`
- unified json: `/Users/tianleli/Desktop/Boulder2/MedMatch-refactor/results/IV_push_17_local_cot_20260413_233315.json`
- legacy json: `/Users/tianleli/Desktop/Boulder2/MedMatch-refactor/scripts/legacy/local/results/IV_push_17_cot_local_20260413_233501.json`
- overall_delta: `0`
- field_delta: `0`
- json_keys_match: `True`
- csv_header_match: `True`

### tier3
- acceptance: `pass`
- unified json: `/Users/tianleli/Desktop/Boulder2/MedMatch-refactor/results/IV_push_17_local_iv_llm_norm_20260413_233724.json`
- legacy json: `/Users/tianleli/Desktop/Boulder2/MedMatch-refactor/scripts/legacy/local/results/IV_push_17_iv_llm_norm_20260413_233802.json`
- raw_overall_delta: `0`
- raw_field_delta: `0`
- norm_overall_delta: `0`
- norm_field_delta: `0`
- json_keys_match: `True`
- csv_header_match: `True`

### exemplar_rag
- acceptance: `pass`
- unified json: `/Users/tianleli/Desktop/Boulder2/MedMatch-refactor/results/IV_push_17_local_rag_20260413_233838.json`
- legacy json: `/Users/tianleli/Desktop/Boulder2/MedMatch-refactor/scripts/legacy/local/results/IV_push_17_rag_20260413_233854.json`
- overall_delta: `0`
- field_delta: `0`
- json_keys_match: `True`
- csv_header_match: `True`

## Notes

- Deltas of `0` indicate parity on the measured slice.
- A `review` status means a schema/layout drift or one-sided execution failure was detected and should be explained before merge.
- A `skipped` status means both unified and legacy paths failed on the same unsupported slice, so the refactor did not introduce a new regression there.
- These notes cover fixed local validation slices only. Add remote parity only if credentials and API access are available.
