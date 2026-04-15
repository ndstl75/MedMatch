#!/usr/bin/env bash
set -euo pipefail

OLLAMA_BIN="${OLLAMA_BIN:-ollama}"
OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11440}"
OLLAMA_MODELS="${OLLAMA_MODELS:-$HOME/.ollama/models}"

if [[ ! -x "$OLLAMA_BIN" ]]; then
  echo "Missing Ollama binary at: $OLLAMA_BIN" >&2
  echo "Download/extract v0.20.6-rc1 first, or set OLLAMA_BIN." >&2
  exit 1
fi

export OLLAMA_HOST
export OLLAMA_MODELS
export GGML_METAL_TENSOR_DISABLE=1

exec "$OLLAMA_BIN" serve
