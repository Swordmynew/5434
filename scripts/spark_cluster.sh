#!/usr/bin/env bash
# Single machine, one worker with 8 cores -> two executors of 4 cores each.
set -euo pipefail
SPARK_HOME="$(python -c 'import os,pyspark; print(os.path.dirname(pyspark.__file__))')"
CLUSTER_DIR="${TASK2_CLUSTER_DIR:-$PWD/runs/cluster}"
mkdir -p "$CLUSTER_DIR"
export SPARK_LOCAL_IP=127.0.0.1
export SPARK_LOCAL_DIRS="${TASK2_SPARK_LOCAL_DIRS:-$CLUSTER_DIR/tmp}"
mkdir -p "$SPARK_LOCAL_DIRS"
if [[ -f "$CLUSTER_DIR/master.pid" || -f "$CLUSTER_DIR/worker.pid" ]]; then
  echo "PID files already exist. Inspect the cluster before starting another one." >&2
  exit 1
fi
nohup "$SPARK_HOME/bin/spark-class" org.apache.spark.deploy.master.Master \
  --host 127.0.0.1 --port 7077 --webui-port 8080 \
  > "$CLUSTER_DIR/master.log" 2>&1 < /dev/null &
echo $! > "$CLUSTER_DIR/master.pid"
# Wait for readiness rather than racing worker registration.
python - <<'PY'
import socket,time
for attempt in range(60):
    try:
        with socket.create_connection(('127.0.0.1',7077),timeout=1): break
    except OSError: time.sleep(1)
else: raise SystemExit('Spark master did not start; inspect master.log')
PY
nohup "$SPARK_HOME/bin/spark-class" org.apache.spark.deploy.worker.Worker \
  --host 127.0.0.1 --cores 8 --memory 20g --webui-port 8081 \
  --work-dir "$CLUSTER_DIR/work" spark://127.0.0.1:7077 \
  > "$CLUSTER_DIR/worker.log" 2>&1 < /dev/null &
echo $! > "$CLUSTER_DIR/worker.pid"
echo "Cluster started. Use --set spark.master=spark://127.0.0.1:7077"
echo "Logs: $CLUSTER_DIR; PID files identify processes for manual shutdown."
