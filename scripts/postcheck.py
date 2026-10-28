"""Inspect a completed run without fitting models or changing blend weights.

Only calibration labels are inspected. Optional reproduction rebuilds test
features/predictions with saved weights, while preserving the submitted CSV.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import pyarrow.parquet as pq


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prediction_csv(path, column="prediction", strict_submission=False):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if strict_submission and reader.fieldnames != ["id", "rating"]:
            raise ValueError(f"Invalid submission columns: {path}")
        if not reader.fieldnames or not {"id", column}.issubset(reader.fieldnames):
            raise ValueError(f"Missing ID/value columns: {path}")
        rows = list(reader)
    ids = [r["id"] for r in rows]
    values = np.asarray([float(r[column]) for r in rows], dtype=float)
    if not ids or len(ids) != len(set(ids)) or not np.isfinite(values).all():
        raise ValueError(f"Empty file, duplicate IDs or non-finite values: {path}")
    return ids, values


def aligned(path, ids):
    found, values = prediction_csv(path)
    if set(found) != set(ids):
        raise ValueError(f"Prediction IDs do not match calibration/test truth: {path}")
    lookup = dict(zip(found, values))
    return np.asarray([lookup[i] for i in ids], dtype=float)


def corr(a, b):
    if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def describe(y, prediction):
    error = prediction - y
    return {"rows": len(y), "rmse": float(np.sqrt(np.mean(error**2))),
            "mae": float(np.mean(np.abs(error))), "bias_prediction_minus_truth": float(np.mean(error)),
            "true_mean": float(np.mean(y)), "prediction_mean": float(np.mean(prediction)),
            "prediction_std": float(np.std(prediction)), "prediction_min": float(np.min(prediction)),
            "prediction_max": float(np.max(prediction)), "prediction_truth_correlation": corr(y, prediction)}


def compare_submissions(reference, candidate, tolerance=1e-6):
    ids, expected = prediction_csv(reference, "rating", True)
    other_ids, actual = prediction_csv(candidate, "rating", True)
    same_order = ids == other_ids
    valid_range = bool(np.all((actual >= 1) & (actual <= 5)))
    difference = float(np.max(np.abs(actual - expected))) if same_order else None
    return {"reference_sha256": sha256(reference), "reproduced_sha256": sha256(candidate),
            "reference_rows": len(ids), "reproduced_rows": len(other_ids),
            "same_ids_and_order": same_order, "reproduced_rating_range_valid": valid_range,
            "byte_identical": Path(reference).read_bytes() == Path(candidate).read_bytes(),
            "absolute_tolerance": tolerance, "maximum_absolute_difference": difference,
            "numerically_reproduced": same_order and valid_range and difference <= tolerance}


def diagnose(cfg, output):
    root = Path(cfg["run_dir"])
    evaluation = read_json(root / "evaluation.json")
    manifest = read_json(root / "manifest.json")
    fields = ["id", "rating", "title", "text_seen", "user_history_count",
              "product_history_count", "parent_history_count"]
    rows = []
    for path in sorted((root / "features" / "calibration").rglob("*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8192, columns=fields):
            rows.extend(batch.to_pylist())
    ids = [str(r["id"]) for r in rows]
    if not ids or len(ids) != len(set(ids)) or len(ids) != manifest["counts"]["calibration"]:
        raise ValueError("Calibration feature rows/IDs do not match the frozen manifest")
    y = np.asarray([r["rating"] for r in rows], dtype=float)
    if not np.isfinite(y).all() or not np.isin(y, [1, 2, 3, 4, 5]).all():
        raise ValueError("Invalid calibration ratings")
    names = evaluation["models"]
    predictions = {name: aligned(root / "predictions" / name / "calibration.csv", ids) for name in names}
    weights = evaluation["selected_weights"]
    if set(weights) - set(names):
        raise ValueError("Selected weights reference missing component models")
    predictions["selected"] = sum(weights.get(name, 0) * predictions[name] for name in names)
    result = {"split": "calibration", "label_count": len(y),
              "note": "No model, calibration transform or blend weight is fitted. Title ratings are hints, not ground truth.",
              "label_counts": {str(k): int(np.sum(y == k)) for k in range(1, 6)},
              "models": {}, "prediction_correlations": {}, "residual_correlations": {}}
    for name, prediction in predictions.items():
        stats = describe(y, prediction)
        stats["by_true_rating"] = {str(k): describe(y[y == k], prediction[y == k]) for k in range(1, 6) if np.any(y == k)}
        stats["by_prediction_bin"] = []
        edges = [-np.inf, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, np.inf]
        for low, high in zip(edges, edges[1:]):
            mask = (prediction >= low) & (prediction < high)
            if mask.any():
                stats["by_prediction_bin"].append({"bin": f"[{low:g},{high:g})", **describe(y[mask], prediction[mask])})
        recorded = evaluation["metrics"]["calibration"].get(name)
        if recorded is not None:
            stats["recorded_rmse"] = recorded
            stats["rmse_matches_record"] = abs(stats["rmse"] - recorded) <= 1e-8
            if not stats["rmse_matches_record"]:
                raise ValueError(f"Saved {name} calibration RMSE does not match evaluation.json")
        result["models"][name] = stats
        result["prediction_correlations"][name] = {n: corr(prediction, p) for n, p in predictions.items()}
        result["residual_correlations"][name] = {n: corr(prediction-y, p-y) for n, p in predictions.items()}
    title_hints = {f"{word} {suffix}": k for k, word in enumerate(["one", "two", "three", "four", "five"], 1)
                   for suffix in ["star", "stars"]}
    buckets = {k: [] for k in range(1, 6)}
    for row in rows:
        title = " ".join((row["title"] or "").casefold().split())
        if title in title_hints:
            buckets[title_hints[title]].append(int(row["rating"]))
    result["rating_title_hints"] = {}
    for k, labels in buckets.items():
        if labels:
            result["rating_title_hints"][str(k)] = {"rows": len(labels),
                "observed_label_counts": {str(j): labels.count(j) for j in range(1, 6)},
                "observed_label_mean": float(np.mean(labels)),
                "fraction_different_from_title_hint": sum(v != k for v in labels) / len(labels)}
    ridge_settings = root / "models" / "ridge" / "settings.json"
    result["ridge_settings"] = read_json(ridge_settings) if ridge_settings.exists() else None
    metadata = list((root / "models" / "ridge" / "pipeline" / "stages").glob("*LinearRegression*/metadata/part-*"))
    result["ridge_saved_parameters"] = [read_json(p).get("paramMap", {}) for p in metadata if p.is_file()]
    exported_fields = [f for f in fields if f != "title"]
    with (output / "calibration_predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=exported_fields + ["pred_" + n for n in predictions])
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow({**{f: row[f] for f in exported_fields},
                             **{"pred_"+n: float(p[index]) for n, p in predictions.items()}})
    sample = Path(cfg["raw_dir"]) / "sampleSubmission.csv"
    with sample.open(encoding="utf-8-sig", newline="") as stream:
        test_ids = [r["id"] for r in csv.DictReader(stream)]
    original_ids, original_values = prediction_csv(root / "submission.csv", "rating", True)
    if original_ids != test_ids or len(test_ids) != len(set(test_ids)) or len(test_ids) != cfg["expected_test_rows"]:
        raise ValueError("Original submission IDs/order/count do not match sampleSubmission.csv")
    test_predictions = {n: aligned(root / "predictions" / n / "test.csv", test_ids) for n in names}
    test_predictions["selected"] = np.clip(sum(weights.get(n, 0)*test_predictions[n] for n in names), 1, 5)
    result["saved_test_predictions_vs_submission_max_abs_difference"] = float(np.max(np.abs(test_predictions["selected"]-original_values)))
    with (output / "test_predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id"] + ["pred_"+n for n in test_predictions])
        writer.writerows([ident] + [float(p[i]) for p in test_predictions.values()] for i, ident in enumerate(test_ids))
    save_json(output / "diagnostics.json", result)
    return result


def run_logged(command, logfile):
    print("RUN:", " ".join(command), flush=True)
    print("LOG:", logfile, flush=True)
    start = time.perf_counter()
    with logfile.open("w", encoding="utf-8") as stream:
        process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
    if process.returncode:
        tail = logfile.read_text(encoding="utf-8", errors="replace").splitlines()[-35:]
        print("\n".join(tail), flush=True)
        raise RuntimeError(f"Command failed with exit {process.returncode}; see {logfile}")
    return time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="runs/main/selected_config.json")
    parser.add_argument("--output-dir")
    parser.add_argument("--archive", default="runs/task2_postcheck.zip")
    parser.add_argument("--spark-master", default="local[8]")
    parser.add_argument("--expected-submission-sha256")
    parser.add_argument("--reproduce", action="store_true")
    parser.add_argument("--benchmark-repeats", type=int, choices=range(0, 11), default=0)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    os.chdir(project)
    output = Path(args.output_dir or ("runs/postcheck_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ"))).resolve()
    output.mkdir(parents=True, exist_ok=False)
    archive = Path(args.archive).resolve()
    if archive.is_relative_to(output):
        raise ValueError("The archive must be outside the output directory")
    status = {"status": "running", "config": args.config, "spark_master_for_checks": args.spark_master,
              "output_dir": str(output), "stages": {}}
    code = 0
    root = None
    reference_hash = None
    try:
        cfg = read_json(args.config)
        root = Path(cfg["run_dir"])
        reference_hash = sha256(root / "submission.csv")
        if args.expected_submission_sha256 and reference_hash != args.expected_submission_sha256:
            raise ValueError("submission.csv differs from the reviewed candidate; original files were not changed")
        original = output / "original"
        original.mkdir()
        small = ["submission.csv", "evaluation.json", "selected_config.json", "manifest.json", "hardware.json",
                 "pilot_selection.json", "timings.jsonl", "parallel_benchmark.json"]
        for name in small:
            if (root / name).is_file():
                shutil.copyfile(root / name, original / name)
        for name in read_json(root / "evaluation.json")["models"]:
            path = root / "predictions" / name / "test.csv"
            if path.is_file():
                target = original / "predictions" / name / "test.csv"
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
        print("1. Calibration diagnostics; no fitting or validation search", flush=True)
        result = diagnose(cfg, output)
        if result["saved_test_predictions_vs_submission_max_abs_difference"] > 1e-8:
            raise ValueError("Saved component predictions/weights no longer reproduce the original submission")
        status["stages"]["diagnostics"] = {"status": "ok", "calibration_rows": result["label_count"]}
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        command = [sys.executable, "-m", "task2", "--config", args.config,
                   "--set", "spark.master=" + args.spark_master]
        if args.reproduce:
            print("2. Reproduce with existing weights (no training)", flush=True)
            candidate = output / "submission_reproduced.csv"
            seconds = run_logged(command + ["predict", "--output", str(candidate)], output / "reproduction.log")
            comparison = compare_submissions(original / "submission.csv", candidate)
            comparison["command_seconds"] = seconds
            comparison["spark_master"] = args.spark_master
            save_json(output / "reproducibility.json", comparison)
            if not comparison["numerically_reproduced"]:
                raise ValueError("Reproduced CSV differs beyond tolerance; keep both files for diagnosis")
            status["stages"]["reproduction"] = {"status": "ok", **comparison}
        measurements = []
        for repeat in range(1, args.benchmark_repeats + 1):
            print(f"3. CPU Spark benchmark repeat {repeat}/{args.benchmark_repeats}", flush=True)
            run_logged(command + ["--set", "device=cpu", "benchmark"], output / f"benchmark_{repeat}.log")
            rows = read_json(root / "parallel_benchmark.json")
            if {r["cores"] for r in rows} != {1, 4, 8}:
                raise ValueError("Incomplete Spark core comparison")
            measurements.append(rows)
            save_json(output / f"benchmark_{repeat}.json", rows)
        if measurements:
            all_rows = [r for trial in measurements for r in trial]
            if len({r["row_count"] for r in all_rows}) != 1 or max(r["rating_sum"] for r in all_rows)-min(r["rating_sum"] for r in all_rows) > 1e-6:
                raise ValueError("CPU benchmark aggregates differ across runs")
            medians = {str(k): statistics.median(r["seconds"] for r in all_rows if r["cores"] == k) for k in [1, 4, 8]}
            summary = {"repeats": len(measurements), "median_seconds": medians,
                       "speedup_vs_one_core": {k: medians["1"]/v for k, v in medians.items()},
                       "row_count": all_rows[0]["row_count"], "spark_cache": "none", "os_page_cache": "uncontrolled",
                       "scope": "Parquet read plus history aggregation; cores and shuffle partitions both vary, not end-to-end training speedup."}
            save_json(output / "benchmark_summary.json", summary)
            status["stages"]["benchmark"] = {"status": "ok", **summary}
        status["status"] = "ok"
    except Exception as exc:
        code = 1
        status["status"] = "failed"
        status["error"] = str(exc)
        (output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print(traceback.format_exc(), flush=True)
    finally:
        if root is not None and reference_hash is not None:
            try:
                status["original_submission_unchanged"] = sha256(root / "submission.csv") == reference_hash
            except OSError:
                status["original_submission_unchanged"] = False
            if not status["original_submission_unchanged"]:
                status["status"], code = "failed", 1
        save_json(output / "status.json", status)
        archive.parent.mkdir(parents=True, exist_ok=True)
        temporary = archive.with_suffix(".tmp")
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as stream:
            for path in sorted(output.rglob("*")):
                if path.is_file():
                    stream.write(path, path.relative_to(output).as_posix())
        temporary.replace(archive)
        print(f"POSTCHECK STATUS: {status['status']}", flush=True)
        print(f"OUTPUT DIRECTORY: {output}", flush=True)
        print(f"UPLOAD THIS FILE: {archive}", flush=True)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
