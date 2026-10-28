"""Configuration, reproducibility, bounded IO, and audit logs."""
from __future__ import annotations

import contextlib
import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import shutil
import subprocess
import time
import zipfile

ENTITY_FIELDS = ("user_id", "prod_id", "parent_prod_id")
ENTITY_PREFIXES = ("user", "product", "parent")
MISSING = "__MISSING_ENTITY__"
BASE_NUMERIC = ["purchased_value", "year_value", "month_sin", "month_cos", "votes_value",
                "title_length_value", "comment_length_value", "title_missing", "comment_missing", "time_missing"]
NUMERIC_NAMES = BASE_NUMERIC + [f"{p}_{s}" for p in ENTITY_PREFIXES
    for s in ["count_value", "mean_value", "prob_1", "prob_2", "prob_3", "prob_4", "prob_5"]]


def load_config(path, overrides=()):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    for override in overrides:
        key, value = override.split("=", 1)
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node[part]
        if parts[-1] not in node:
            raise ValueError(f"Unknown configuration key: {key}")
        node[parts[-1]] = value
    if cfg["split"]["folds"] < 2 or cfg["features"]["smoothing"] <= 0:
        raise ValueError("Need >=2 cross-fitting folds and positive smoothing")
    return cfg


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def clean_text(value):
    # Decode first, including escaped tags, while preserving sentiment punctuation.
    value = html.unescape(value or "")
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", value)).strip()


def largest_remainder(counts, size):
    """Exact stratified quotas; ties are resolved by rating for stable splits."""
    total = sum(counts.values())
    if not 0 <= size <= total:
        raise ValueError(f"Requested {size} rows from {total}")
    raw = {k: size * v / total for k, v in counts.items()}
    quota = {k: math.floor(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: (-(raw[k] - quota[k]), k))[:size - sum(quota.values())]:
        quota[k] += 1
    return quota


def seed_everything(seed):
    random.seed(seed)
    import numpy as np
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def device_for(cfg):
    import torch
    value = cfg["device"]
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(value)


def synchronize():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


@contextlib.contextmanager
def timed(cfg, stage, **metadata):
    synchronize()
    start = time.perf_counter()
    status = "ok"
    try:
        yield
    except BaseException:
        status = "failed"
        raise
    finally:
        synchronize()
        row = {"stage": stage, "seconds": time.perf_counter() - start,
               "status": status, "unix_time": time.time(), **metadata}
        root = Path(cfg["run_dir"])
        root.mkdir(parents=True, exist_ok=True)
        with (root / "timings.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row), flush=True)


def hardware_info():
    import psutil
    info = {"platform": platform.platform(), "python": platform.python_version(),
            "cpu_model": platform.processor(), "logical_cpus": os.cpu_count(),
            "physical_memory_bytes": psutil.virtual_memory().total, "servers": 1}
    if hasattr(os, "sched_getaffinity"):
        info["cpu_affinity_count"] = len(os.sched_getaffinity(0))
    # Rented containers can expose the host totals; record cgroup limits too.
    for path in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        value = Path(path)
        if value.exists():
            text = value.read_text().strip()
            if text.isdigit():
                info["container_memory_limit_bytes"] = int(text)
                break
    quota_path = Path("/sys/fs/cgroup/cpu.max")
    if quota_path.exists():
        quota, period = quota_path.read_text().split()
        if quota != "max":
            info["container_cpu_quota"] = int(quota) / int(period)
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                info["cpu_model"] = line.split(":", 1)[1].strip()
                break
    try:
        import torch
        info.update(torch=torch.__version__, cuda=torch.version.cuda,
                    torch_cpu_threads=torch.get_num_threads(),
                    gpus=[{"name": torch.cuda.get_device_name(i),
                           "memory_bytes": torch.cuda.get_device_properties(i).total_memory}
                          for i in range(torch.cuda.device_count())])
    except ImportError:
        info["torch"] = "not installed"
    info["java"] = shutil.which("java")
    if info["java"]:
        result = subprocess.run([info["java"], "-version"], capture_output=True, text=True)
        info["java_version"] = (result.stderr or result.stdout).strip()
    return info


def stage_raw(cfg):
    """Extract only the three expected entries, never trusting archive paths."""
    root = Path(cfg["raw_dir"])
    root.mkdir(parents=True, exist_ok=True)
    names = ["train.csv", "test.csv", "sampleSubmission.csv"]
    if all((root / n).is_file() for n in names):
        return root
    with zipfile.ZipFile(cfg["archive"]) as archive:
        for name in names:
            candidates = [x for x in archive.infolist() if Path(x.filename).name == name]
            if len(candidates) != 1:
                raise ValueError(f"Expected exactly one {name} in archive")
            item = candidates[0]
            destination = root / name
            if destination.exists() and destination.stat().st_size == item.file_size:
                continue
            temporary = destination.with_suffix(".csv.partial")
            with archive.open(item) as source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
            os.replace(temporary, destination)
    return root


def parquet_files(path):
    path = Path(path)
    if path.is_file():
        return [path] if path.suffix == ".parquet" else []
    return sorted(p for p in path.rglob("*.parquet") if not p.name.startswith("."))


def iter_rows(path, columns=None, batch_size=2048):
    import pyarrow.parquet as pq
    for file in parquet_files(path):
        for batch in pq.ParquetFile(file).iter_batches(batch_size=batch_size, columns=columns):
            yield from batch.to_pylist()


def rmse(truth, prediction):
    import numpy as np
    truth, prediction = np.asarray(truth, dtype=float), np.asarray(prediction, dtype=float)
    if len(truth) == 0 or truth.shape != prediction.shape or not np.isfinite(prediction).all() or not np.isfinite(truth).all():
        raise ValueError("RMSE requires aligned, finite, non-empty arrays")
    return float(np.sqrt(np.mean((truth - prediction) ** 2)))


def write_predictions(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "prediction"])
        writer.writeheader()
        seen = set()
        for record in records:
            ident, value = str(record["id"]), float(record["prediction"])
            if ident in seen or not math.isfinite(value):
                raise ValueError(f"Duplicate id or non-finite prediction: {ident}")
            seen.add(ident)
            writer.writerow({"id": ident, "prediction": format(min(5., max(1., value)), ".10f")})


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
