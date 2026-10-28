"""Reproduce the scored text-only + frozen postprocessor candidate.

Full mode rebuilds raw-test features and runs the saved text weights. Cached
mode only verifies CSV export and is explicitly recorded as such. No fitting,
new training, weight selection or automatic Kaggle submission is performed.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from zipfile import ZipFile, ZIP_DEFLATED


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_bytes(ids, values):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(["id", "rating"])
    writer.writerows((ident, format(value, ".10f")) for ident, value in zip(ids, values))
    return stream.getvalue().encode("utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="runs/main/selected_config.json")
    parser.add_argument("--postprocessor", default="configs/best_postprocess.json")
    parser.add_argument("--spark-master", default="local[8]")
    parser.add_argument("--output-dir")
    parser.add_argument("--archive", default="runs/task2_best_repro_results.zip")
    parser.add_argument("--sample")
    parser.add_argument("--cached-predictions", help="CSV-export check only; does not verify model inference")
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parents[1])
    destination = Path(args.output_dir or ("runs/best_repro_"+datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ"))).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    archive = Path(args.archive).resolve()
    if archive.is_relative_to(destination):
        raise ValueError("Archive must be outside the result directory")
    mode = "cached_csv_export_only" if args.cached_predictions else "full_raw_test_inference"
    result = {"status": "running", "execution_mode": mode, "output_directory": str(destination)}
    code, root, preserved = 0, None, {}
    inference_started = None
    started = time.perf_counter()
    try:
        cfg, params = read_json(args.config), read_json(args.postprocessor)
        if params["source_model"] != "text" or params["transform"] != "uniform_label_mixture_inverse_v1":
            raise ValueError("Unsupported frozen postprocessor")
        rho = float(params["rho"])
        if not math.isfinite(rho) or not 0 <= rho < 1 or params["uniform_rating_mean"] != 3 or params["rating_bounds"] != [1., 5.]:
            raise ValueError("Invalid frozen postprocessor parameters")
        root = Path(cfg["run_dir"])
        shutil.copyfile(args.config, destination / "source_config.json")
        shutil.copyfile(args.postprocessor, destination / "frozen_postprocessor.json")
        if not args.cached_predictions:
            for name in ["submission.csv", "evaluation.json", "selected_config.json"]:
                if (root / name).is_file():
                    preserved[name] = sha256(root / name)
            manifest = read_json(root / "manifest.json")
            for key in ["features", "split"]:
                if cfg[key] != manifest["config"][key]:
                    raise ValueError(f"Frozen preprocessing settings changed: {key}")
            cfg["spark"]["master"] = args.spark_master
            os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["PYSPARK_PYTHON"] = os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
            from task2.common import synchronize
            synchronize()
            inference_started = time.perf_counter()
            from task2.spark_features import prepare_test
            from task2.training import predict
            print("Rebuilding test features from raw CSV", flush=True)
            prepare_test(cfg)
            print("Predicting with saved text checkpoint", flush=True)
            predict(cfg, "text")
        prediction_path = Path(args.cached_predictions) if args.cached_predictions else root / "predictions" / "text" / "test.csv"
        sample_path = Path(args.sample) if args.sample else Path(cfg["raw_dir"]) / "sampleSubmission.csv"
        with sample_path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != ["id", "rating"]:
                raise ValueError("Unexpected sample schema")
            ids = [r["id"] for r in reader]
        with prediction_path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != ["id", "prediction"]:
                raise ValueError("Unexpected component prediction schema")
            rows = list(reader)
        lookup = {r["id"]: float(r["prediction"]) for r in rows}
        if len(ids) != cfg["expected_test_rows"] or len(ids) != len(set(ids)) or len(rows) != len(lookup) or set(lookup) != set(ids):
            raise ValueError("Prediction/sample ID, order or count check failed")
        raw = [lookup[i] for i in ids]
        if not all(math.isfinite(v) and 1 <= v <= 5 for v in raw):
            raise ValueError("Non-finite or invalid saved text predictions")
        base_sha = hashlib.sha256(csv_bytes(ids, raw)).hexdigest()
        corrected = [min(5., max(1., (v-3*rho)/(1-rho))) for v in raw]
        candidate = destination / "submission_text_noise_hypothesis.csv"
        candidate.write_bytes(csv_bytes(ids, corrected))
        if inference_started is not None:
            synchronize()
            result["full_inference_seconds"] = time.perf_counter()-inference_started
            result["inference_timing_scope"] = "Raw test feature generation, text model loading/prediction, frozen postprocessing and CSV output; synchronized at both ends."
        actual_sha = sha256(candidate)
        result.update({"rows": len(ids), "rho": rho, "source_model": "text", "reference_public_rmse": params["user_reported_public_rmse"],
            "baseline_text_submission_sha256": base_sha,
            "baseline_text_hash_matches": base_sha == params["baseline_text_submission_sha256"],
            "expected_submission_sha256": params["expected_submission_sha256"], "actual_submission_sha256": actual_sha,
            "matches_scored_submission_sha256": actual_sha == params["expected_submission_sha256"],
            "rating_minimum": min(corrected), "rating_maximum": max(corrected),
            "prediction_file_sha256": sha256(prediction_path), "sample_file_sha256": sha256(sample_path),
            "spark_master": args.spark_master if not args.cached_predictions else None})
        if not result["matches_scored_submission_sha256"] or not result["baseline_text_hash_matches"]:
            raise ValueError("Prediction differs from the scored candidate; output kept for investigation")
        if not args.cached_predictions:
            truth_path = root / "features" / "validation"
            validation_path = root / "predictions" / "text" / "validation.csv"
            if truth_path.is_dir() and validation_path.is_file():
                from task2.common import iter_rows, rmse
                from task2.ensemble import read_predictions
                truth_rows = list(iter_rows(truth_path, ["id", "rating"]))
                val_ids = [r["id"] for r in truth_rows]
                val_truth = [r["rating"] for r in truth_rows]
                val_predictions = read_predictions(validation_path)
                if len(val_ids) != len(set(val_ids)) or len(val_ids) != manifest["counts"]["validation"] or set(val_ids) != set(val_predictions):
                    raise ValueError("Saved validation predictions do not match the frozen validation IDs")
                val_raw = [val_predictions[i] for i in val_ids]
                val_corrected = [min(5., max(1., (v-3*rho)/(1-rho))) for v in val_raw]
                recorded = read_json(root / "evaluation.json")["metrics"]["validation"]["text"]
                val_result = {"status": "ok", "rows": len(val_ids), "raw_text_rmse": rmse(val_truth, val_raw),
                    "postprocessed_rmse_on_observed_labels": rmse(val_truth, val_corrected),
                    "recorded_raw_text_rmse": recorded,
                    "scope": "One audit of the frozen postprocessor against existing validation labels/predictions; no parameter fitting or checkpoint selection."}
                save_json(destination / "observed_validation_metrics.json", val_result)
                if abs(val_result["raw_text_rmse"]-recorded) > 1e-8:
                    raise ValueError("Saved validation RMSE does not match the original text checkpoint evaluation")
                result["observed_validation_metrics"] = val_result
            else:
                result["observed_validation_metrics"] = {"status": "not_available", "note": "Inference hash check passed; saved validation artifacts not supplied."}
        result["status"] = "ok"
    except Exception as exc:
        code = 1
        result.update(status="failed", error=str(exc))
        (destination / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print(traceback.format_exc(), flush=True)
    finally:
        if not args.cached_predictions and root is not None:
            result["original_artifacts_unchanged"] = {name: (root/name).is_file() and sha256(root/name) == value for name, value in preserved.items()}
            if not all(result["original_artifacts_unchanged"].values()):
                result["status"], code = "failed", 1
            common = sys.modules.get("task2.common")
            if common is not None:
                try:
                    common.synchronize()
                except Exception as exc:
                    result["synchronization_error"] = str(exc)
                    result["status"], code = "failed", 1
        result["elapsed_seconds"] = time.perf_counter()-started
        save_json(destination / "reproduction.json", result)
        archive.parent.mkdir(parents=True, exist_ok=True)
        temporary = archive.with_suffix(".tmp")
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as stream:
            for file in sorted(destination.rglob("*")):
                if file.is_file():
                    stream.write(file, file.relative_to(destination).as_posix())
        temporary.replace(archive)
        print("BEST REPRO STATUS:", result["status"], flush=True)
        print("MODE:", mode, flush=True)
        print("UPLOAD RESULT:", archive, flush=True)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
