# Remote Parity Follow-Up

This note captures the remote side-by-side follow-up for the refactor branch using the smallest comparable archived slice that was available locally.

- comparison method: refactor remote smoke output versus the first matching medication row from archived legacy remote result files
- comparable medications:
  - `IV push`: `Lacosamide`
  - `IV intermittent`: `Amiodarone`
- why first-row slices were used:
  - the archived legacy remote artifacts available locally were full-dataset runs
  - the refactor remote validation was intentionally a 1-entry smoke slice
  - the first medication in each archived legacy file matched the refactor smoke medication, so that row gives a like-for-like comparison

## Results

### CoT IV push

- slice: `Lacosamide`
- overall_delta: `+1`
- field_delta: `+1`
- json_keys_match: `True`
- csv_header_match: `True`

### CoT IV intermittent

- slice: `Amiodarone`
- overall_delta: `0`
- field_delta: `0`
- json_keys_match: `True`
- csv_header_match: `True`

### Tier 3 IV push

- slice: `Lacosamide`
- raw_overall_delta: `0`
- raw_field_delta: `0`
- norm_overall_delta: `-1`
- norm_field_delta: `-1`
- json_keys_match: `True`
- csv_header_match: `True`

### Tier 3 IV intermittent

- slice: `Amiodarone`
- raw_overall_delta: `0`
- raw_field_delta: `0`
- norm_overall_delta: `0`
- norm_field_delta: `0`
- json_keys_match: `True`
- csv_header_match: `True`

### Exemplar-RAG IV push

- slice: `Lacosamide`
- overall_delta: `0`
- field_delta: `0`
- json_keys_match: `True`
- csv_header_match: `True`

## Notes

- This note closes the main remote parity gap for the follow-up task without requiring a fresh API run.
- The only observed drift on the comparable remote Tier 3 slice is `IV push` normalized output: the refactor smoke slice is lower by `1` field and `1` all-correct result on `Lacosamide`.
- That matches the earlier remote smoke observation that `IV push` Tier 3 remote normalization is the main remaining behavior to watch.
- No comparable archived single-entry legacy remote baseline artifact was available locally, so baseline is not included in this note.
- `scripts/run_parity.py` now supports `--backend remote` for future live reruns when a current `GOOGLE_API_KEY` is present in the environment or local `.env`.
