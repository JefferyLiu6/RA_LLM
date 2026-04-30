#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

GEN_COUNT="${GEN_COUNT:-532}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-25}"
GEN_MODEL="${GEN_MODEL:-gpt-4o-mini}"
LOCAL_GEN_MODEL="${LOCAL_GEN_MODEL:-codex_session}"

echo "== Dataset pipeline =="
echo "Python: $PYTHON_BIN"
echo

echo "1/4 Convert original 68 examples into metadata-rich gold records"
"$PYTHON_BIN" src/dataset_pipeline.py gold

echo
if [[ -n "${OPENAI_API_KEY:-}" ]]; then
  echo "2/4 OPENAI_API_KEY is set; generating synthetic records"
  "$PYTHON_BIN" src/dataset_pipeline.py generate-openai \
    --count "$GEN_COUNT" \
    --batch-size "$GEN_BATCH_SIZE" \
    --model "$GEN_MODEL"
else
  echo "2/4 OPENAI_API_KEY is not set; generating local synthetic records"
  "$PYTHON_BIN" src/dataset_pipeline.py generate-local \
    --count "$GEN_COUNT" \
    --model "$LOCAL_GEN_MODEL" \
    --no-append
fi

echo
echo "3/4 Build fixed SFT split files"
"$PYTHON_BIN" src/dataset_pipeline.py build

echo
echo "4/4 Validate dataset"
"$PYTHON_BIN" src/dataset_pipeline.py validate

echo
echo "Dataset files:"
wc -l data/dataset_gold.jsonl data/sft_all.jsonl data/sft_train.jsonl data/sft_val.jsonl data/sft_test.jsonl
