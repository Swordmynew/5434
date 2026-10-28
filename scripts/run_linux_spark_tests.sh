#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv-linux/bin/activate
export JAVA_HOME="$(python -c 'import jdk4py; print(jdk4py.JAVA_HOME)')"
export PATH="$JAVA_HOME/bin:$PATH"
export PYSPARK_PYTHON="$(command -v python)"
export PYSPARK_DRIVER_PYTHON="$PYSPARK_PYTHON"
export SPARK_LOCAL_IP=127.0.0.1
python -m pytest -q tests/test_spark.py --tb=short
