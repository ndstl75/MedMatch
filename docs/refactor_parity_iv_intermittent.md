# Refactor Parity Note

- backend: `local`
- category: `iv_intermittent`
- python: `/opt/anaconda3/envs/medmatch/bin/python3`
- ollama model: `medgemma:27b`

## Results

### baseline
- acceptance: `pass`
- unified json: `/Users/tianleli/Desktop/Boulder2/MedMatch-refactor/results/IV_intermittent_16_local_baseline_20260413_235155.json`
- legacy json: `/Users/tianleli/Desktop/Boulder2/MedMatch-refactor/scripts/legacy/local/results/IV_intermittent_16_dataset_aligned_appendix_style_20260413_235209.json`
- overall_delta: `0`
- field_delta: `0`
- json_keys_match: `True`
- csv_header_match: `True`

### cot
- acceptance: `pass`
- unified json: `/Users/tianleli/Desktop/Boulder2/MedMatch-refactor/results/IV_intermittent_16_local_cot_20260413_235218.json`
- legacy json: `/Users/tianleli/Desktop/Boulder2/MedMatch-refactor/scripts/legacy/local/results/IV_intermittent_16_cot_local_20260413_235335.json`
- overall_delta: `0`
- field_delta: `0`
- json_keys_match: `True`
- csv_header_match: `True`

### tier3
- acceptance: `pass`
- unified json: `/Users/tianleli/Desktop/Boulder2/MedMatch-refactor/results/IV_intermittent_16_local_iv_llm_norm_20260413_235457.json`
- legacy json: `/Users/tianleli/Desktop/Boulder2/MedMatch-refactor/scripts/legacy/local/results/IV_intermittent_16_iv_llm_norm_20260413_235530.json`
- raw_overall_delta: `0`
- raw_field_delta: `0`
- norm_overall_delta: `0`
- norm_field_delta: `0`
- json_keys_match: `True`
- csv_header_match: `True`

### exemplar_rag
- acceptance: `skipped`
- reason: `both unified and legacy entrypoints failed on this slice`
- unified failure: `KeyError: 'IV intermittent (16)'`
- legacy failure: `No new result pair detected in /Users/tianleli/Desktop/Boulder2/MedMatch-refactor/scripts/legacy/local/results for prefix IV_intermittent_16`

## Notes

- Deltas of `0` indicate parity on the measured slice.
- A `review` status means a schema/layout drift or one-sided execution failure was detected and should be explained before merge.
- A `skipped` status means both unified and legacy paths failed on the same unsupported slice, so the refactor did not introduce a new regression there.
- These notes cover fixed local validation slices only. Add remote parity only if credentials and API access are available.
