#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv-server/bin/activate
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export PATH="$JAVA_HOME/bin:$PATH"
export PYTHONUNBUFFERED=1
export TMPDIR="$PWD/.tmp"
export HF_HOME="$PWD/.hf-cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export SPARK_LOCAL_IP=127.0.0.1
export SPARK_LOCAL_DIRS="$PWD/.tmp/spark"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYSPARK_PYTHON="$(command -v python)"
export PYSPARK_DRIVER_PYTHON="$PYSPARK_PYTHON"
mkdir -p "$TMPDIR" "$SPARK_LOCAL_DIRS" runs
python -u scripts/reproduce_best.py \
    --config runs/main/selected_config.json \
    --postprocessor configs/best_postprocess.json \
    --spark-master 'local[8]' \
    2>&1 | tee runs/best_repro.log
