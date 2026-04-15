# MedMatch Local Gemma4

这个子目录是 `MedMatch` 的本地模型版，保留原来的 prompt、评估、打分逻辑，只把云端 `google.genai` 调用改成了本地 `Ollama` API。

默认配置：

- `OLLAMA_BASE_URL=http://localhost:11434`
- `OLLAMA_MODEL=gemma4:e4b`

这台机器当前验证通过的组合：

- Ollama `0.20.6-rc1`
- `GGML_METAL_TENSOR_DISABLE=1`

原因：

- 默认配置下，`gemma4:e4b` 在这台 `Apple M5 Pro` 上会触发 Metal tensor 编译错误
- 关闭 `GGML` 的 Metal tensor path 后，本地 API 可以正常返回

当前推荐入口：

- `../scripts/run_baseline.py --backend local`
- `../scripts/run_cot.py --backend local`
- `../scripts/run_normalization.py --backend local`

迁移期保留的本地脚本现在位于 `scripts/legacy/local/`：

- `medmatch_test_local_appendix_exact.py`：本地 baseline / 全 5 类评估
- `iv_cot_experiment_local.py`：IV intermittent / IV push 的两步 CoT 实验
- `iv_llm_normalize_local.py`：IV intermittent / IV push 的两步 LLM normalization
- `iv_llm_normalize_conditional_local.py`：IV intermittent / IV push 的 conditional normalization
- `medmatch_test_local.py`
- `medmatch_single_test_local.py`

现在也有统一的单条调试入口：

- `scripts/run_single.py --backend local --category ... --prompt "..."`
- `oral_llm_normalize_local.py`
- `oral_strict_format_experiment_local.py`

说明：

- oral normalization 主要覆盖 `PO Solid` 和 `PO liquid`
- IV normalization 主要覆盖 `IV intermittent` 和 `IV push`
- `IV continuous` 目前保留为 baseline / audit 对照，暂不建议继续做 normalization-only 优化

常用示例：

```bash
cd "$(git rev-parse --show-toplevel)"
python3 scripts/run_baseline.py --backend local --category iv_push
```

```bash
cd "$(git rev-parse --show-toplevel)"
MEDMATCH_NUM_RUNS=1 python3 scripts/run_baseline.py --backend local --category all
```

如果你要换本地模型：

```bash
export OLLAMA_MODEL="gemma4:26b"
```

如果 Ollama 不在默认地址：

```bash
export OLLAMA_BASE_URL="http://127.0.0.1:11434"
```

当前机器推荐启动方式：

```bash
cd "$(git rev-parse --show-toplevel)"
bash scripts/local/start_ollama_gemma4_workaround.sh
```

或者手动：

```bash
export GGML_METAL_TENSOR_DISABLE=1
export OLLAMA_HOST="127.0.0.1:11440"
export OLLAMA_BIN="/path/to/ollama"
bash scripts/local/start_ollama_gemma4_workaround.sh
```

然后另一个终端里：

```bash
cd "$(git rev-parse --show-toplevel)"
OLLAMA_BASE_URL="http://127.0.0.1:11440" python3 scripts/legacy/local/medmatch_single_test_local.py
```

或使用统一入口：

```bash
python3 scripts/run_single.py --backend local --category iv_push --prompt "Famotidine 20 mg, 2 mL of a 20 mg/2 mL vial, was administered twice daily via intravenous push."
```

说明：

- 数据集文件会优先在当前子目录找；找不到时会自动回退到上一级 `MedMatch` 根目录。
- 结果文件会写到这个子目录下的 `results/`。
- 如果你改回默认配置后又看到 `model failed to load`，优先确认是否仍然带着 `GGML_METAL_TENSOR_DISABLE=1`。
