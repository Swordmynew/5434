#!/usr/bin/env bash
# Run after the first submission; no model fitting or weight selection.
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
python -u scripts/postcheck.py \
    --config runs/main/selected_config.json \
    --spark-master 'local[8]' \
    --expected-submission-sha256 5d6f585a71b5ff66a889a4237afaa4115a01e325fa3f547c0e3baf3898b2079f \
    --reproduce --benchmark-repeats 3 \
    2>&1 | tee runs/postcheck.log
