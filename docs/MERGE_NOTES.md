# Merge Notes

## Fork Point

- Lab-aligned Stage 1 base: `f732e15`
- Stage 2 branch: `stage2/integrate-into-lab-pipeline`
- Safety tag before refactor: `pre-stage2`

## Stage 1 Summary

- Centralized CoT and normalization prompt text in `src/prompt_medmatch.py`
- Removed duplicated prompt constants from the old `src/medmatch/experiments/*` runners
- Verified prompt-builder imports and CLI parsing before Stage 2

## Stage 2 Summary

- Made `scripts/probing_medmatch.py` the canonical MedMatch runner for:
  - baseline prompting: `zero`, `few`, `one_shot`
  - CoT prompting: `cot`
  - normalization prompting: `normalization`
- Kept lab CSVs in `data/med_match/` as the canonical benchmark source
- Removed the redundant baseline stack:
  - `scripts/run_baseline.py`
  - `src/medmatch/core/paper_baseline.py`
  - `src/medmatch/experiments/`
- Kept compatibility shims:
  - `scripts/run_cot.py`
  - `scripts/run_normalization.py`
  - `scripts/run_tier3.py`
- Kept `scripts/run_single.py` as a debug helper

## Parity Checks

Create the old-path worktree once:

```bash
git worktree add ../MedMatch-pre-stage2 pre-stage2
```

Run the Stage 2 parity test:

```bash
PYTHONPATH=src:. python -m unittest tests.test_stage2_parity
```

CLI sanity checks:

```bash
PYTHONPATH=src:. python scripts/probing_medmatch.py --help
PYTHONPATH=src:. python scripts/run_cot.py --help
PYTHONPATH=src:. python scripts/run_normalization.py --help
PYTHONPATH=src:. python scripts/run_single.py --help
```

Tiny local smokes:

```bash
OLLAMA_MODEL='gemma4:e4b' PYTHONPATH=src:. python scripts/probing_medmatch.py --mode local --model_name gemma4:e4b --prompting_type cot --num_runs 1 --subset_size 1
OLLAMA_MODEL='gemma4:e4b' PYTHONPATH=src:. python scripts/probing_medmatch.py --mode local --model_name gemma4:e4b --prompting_type normalization --num_runs 1 --subset_size 1
```
