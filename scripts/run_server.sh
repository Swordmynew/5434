#!/usr/bin/env bash
# Launch only after installing dependencies using README.md.
set -euo pipefail
cd "$(dirname "$0")/.."
case "${1:-}" in
  --pretrained-approved) MODEL_OPTIONS=(--set text.pretrained_allowed=true) ;;
  --scratch) MODEL_OPTIONS=(--set text.mode=scratch) ;;
  *) echo "Usage: bash scripts/run_server.sh --pretrained-approved|--scratch" >&2; exit 2 ;;
esac
mkdir -p .tmp .hf-cache runs
export TMPDIR="$PWD/.tmp"
export HF_HOME="$PWD/.hf-cache"
export SPARK_LOCAL_DIRS="$PWD/.tmp/spark"
mkdir -p "$SPARK_LOCAL_DIRS"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYSPARK_PYTHON="$(command -v python)"
export PYSPARK_DRIVER_PYTHON="$PYSPARK_PYTHON"
export SPARK_LOCAL_IP=127.0.0.1
python -c 'import torch; assert torch.cuda.is_available(), "CUDA GPU unavailable"; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
python -m task2 doctor
bash scripts/spark_cluster.sh
python -m task2 --set spark.master=spark://127.0.0.1:7077 "${MODEL_OPTIONS[@]}" run
python -m pip freeze > runs/main/environment.txt
