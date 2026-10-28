"""Unified entry point. All heavy imports are delayed until their command runs."""
from __future__ import annotations

import argparse
import copy
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time

from task2.common import hardware_info, load_config, read_json, timed, write_json


def doctor(cfg):
    info = hardware_info()
    run = Path(cfg["run_dir"])
    run.mkdir(parents=True, exist_ok=True)
    info["free_disk_gb"] = shutil.disk_usage(run).free / 1024**3
    info["configured_spark"] = cfg["spark"]
    info["packages"] = {}
    for name in ["torch", "numpy", "pyarrow", "pyspark", "transformers", "tokenizers", "sentencepiece", "scipy", "pandas", "psutil"]:
        try:
            info["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            info["packages"][name] = "MISSING"
    write_json(run / "hardware.json", info)
    print(json.dumps(info, ensure_ascii=False, indent=2))
    if not info["java"]:
        raise RuntimeError("Java 17 required: install openjdk-17-jdk and set JAVA_HOME/PATH")
    missing = [name for name, value in info["packages"].items() if value == "MISSING"]
    if missing:
        raise RuntimeError("Missing runtime packages: " + ", ".join(missing))
    if info["free_disk_gb"] < cfg["minimum_free_gb"]:
        raise RuntimeError("Insufficient free disk on run_dir filesystem")
    return info


def predict_pipeline(cfg, output=None):
    from task2 import baseline, ensemble
    from task2.spark_features import prepare_test, session
    from task2.common import write_predictions
    root = Path(cfg["run_dir"])
    selection = read_json(root / "evaluation.json")["selected_weights"]
    with timed(cfg, "total_inference", includes="raw test -> features -> model loading -> prediction -> CSV"):
        spark = session(cfg, "task2-complete-inference")
        try:
            prepare_test(cfg, spark)
            for name, weight in selection.items():
                if weight <= 0:
                    continue
                if name == "mean":
                    mean = read_json(root / "manifest.json")["mean"]
                    rows = spark.read.parquet(str(root / "features" / "test")).select("id").toLocalIterator()
                    write_predictions(root / "predictions" / "mean" / "test.csv", ({"id": r["id"], "prediction": mean} for r in rows))
                elif (root / "models" / name / "pipeline").exists():
                    baseline.predict(cfg, name, spark)
        finally:
            spark.stop()
        for name, weight in selection.items():
            if weight > 0 and (root / "models" / name / "training_config.json").exists():
                from task2 import training
                training.predict(cfg, name)
        output = ensemble.submission(cfg, output)
        print(f"Submission: {output.resolve()}")
    return output


def benchmark(cfg):
    from pyspark.sql import functions as F
    from task2.spark_features import session
    results = []
    # Identical Parquet input, no Spark persist/cache, force full aggregate action.
    for cores in [1, 4, 8]:
        spec = copy.deepcopy(cfg)
        spec["spark"].update(master=f"local[{cores}]", partitions=cores*4, cores_max=cores)
        spark = session(spec, f"task2-benchmark-{cores}")
        try:
            start = time.perf_counter()
            data = spark.read.parquet(str(Path(cfg["run_dir"]) / "clean"))
            summary = data.groupBy("user_id", "prod_id", "parent_prod_id").agg(F.count("rating").alias("n"), F.sum("rating").alias("s"))
            totals = summary.agg(F.sum("n"), F.sum("s")).first()
            results.append({"cores": cores, "seconds": time.perf_counter()-start, "row_count": totals[0], "rating_sum": totals[1],
                            "stage": "parquet_read_and_history_aggregation", "spark_cache": "none", "os_cache": "uncontrolled"})
        finally:
            spark.stop()
    write_json(Path(cfg["run_dir"]) / "parallel_benchmark.json", results)


def report(cfg):
    root = Path(cfg["run_dir"])
    evaluation = read_json(root / "evaluation.json")
    hardware = read_json(root / "hardware.json") if (root / "hardware.json").exists() else hardware_info()
    lines = ["# Task 2 实验结果", "", "以下数值从本次运行工件读取；不包含未执行的实验。", "", "## 数据与硬件", "", "```json",
             json.dumps({"counts": read_json(root / "manifest.json")["counts"], "hardware": hardware}, ensure_ascii=False, indent=2), "```",
             "", "## 评分结果", "", "| 划分 | 模型 | RMSE |", "|---|---|---|"]
    for split, values in evaluation["metrics"].items():
        lines += [f"| {split} | {name} | {value:.6f} |" for name, value in values.items()]
    lines += ["", "## 融合与分层结果", "", "```json", json.dumps(evaluation, ensure_ascii=False, indent=2), "```", "", "## 训练轮次计时", "", "```json"]
    epochs = {str(p.relative_to(root)): read_json(p) for p in sorted((root / "models").glob("*/epoch_*.json"))}
    lines += [json.dumps(epochs, ensure_ascii=False, indent=2), "```", "", "## 阶段计时", "", "| 阶段 | 模型 | 状态 | 秒 |", "|---|---|---|---|"]
    if (root / "timings.jsonl").exists():
        for line in (root / "timings.jsonl").read_text().splitlines():
            row = json.loads(line)
            lines.append(f"| {row['stage']} | {row.get('model', '')} | {row['status']} | {row['seconds']:.3f} |")
    lines += ["", "总离线时间以完整 `run` 命令记录为准；分开执行的命令不伪装成一次端到端测量。部分轮次不计为完整 epoch。",
              "", "## 人工补充", "", "填写队名、姓名、学号，解释设计和消融结果，加入参考文献，以及合计 100% 的贡献表。"]
    (root / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def fixture(destination):
    """Synthetic data including quotes/newlines, missing body and cold-start IDs."""
    root = Path(destination)
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    fields = ["id", "title", "user_id", "comment", "prod_id", "parent_prod_id", "time", "votes", "purchased"]
    rows = []
    sentiments = ["terrible broken", "poor disappointing", "okay average", "good useful", "excellent perfect"]
    for i in range(600):
        rating = i % 5 + 1
        rows.append({"id": str(i), "title": f"{sentiments[rating-1]} review {i%7}", "comment": "" if i % 37 == 0 else f'{sentiments[rating-1]}, says "reviewer".\nSecond line<br />works &amp; lasts.',
                     "user_id": f"u{i%19}", "prod_id": f"p{i%23}", "parent_prod_id": f"f{i%9}",
                     "time": str(1600000000000+i*100000), "votes": str(i%8), "purchased": "True" if i%3 else "False", "rating": rating})
    test = [{k: v for k, v in rows[i].items() if k != "rating"} for i in range(20)]
    test[-1]["user_id"] = "never-seen-user"
    test[-1]["prod_id"] = "never-seen-product"
    test[-1]["parent_prod_id"] = "never-seen-parent"
    for name, data, columns in [("train.csv", rows, fields+["rating"]), ("test.csv", test, fields),
                                 ("sampleSubmission.csv", [{"id": r["id"], "rating": 3} for r in reversed(test)], ["id", "rating"])]:
        with (raw / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(data)
    cfg = load_config("configs/server.json")
    cfg.update(raw_dir=str(raw.resolve()), run_dir=str((root / "run").resolve()), expected_test_rows=20, minimum_free_gb=0, device="cpu")
    cfg["spark"].update(master="local[2]", partitions=2, driver_memory="2g", executor_memory="1g", cores_max=2, executor_cores=1)
    cfg["split"].update(calibration_size=50, validation_size=50)
    cfg["baseline"].update(hash_features=1024, max_iter=8)
    cfg["text"].update(mode="scratch", max_length=48, title_tokens=12, workers=0, tokenizer_workers=1,
                         batch_size=16, accumulation=2, epochs=1, amp=False, gradient_checkpointing=False,
                         scratch_hidden=32, scratch_layers=1, scratch_vocab_size=256, evaluate_every_steps=0, max_hours=0.2)
    cfg["relation"].update(batch_size=64, workers=0, epochs=2, max_hours=0.2)
    cfg["pilot"]["enabled"] = False
    write_json(root / "config.json", cfg)
    print(root / "config.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/server.json")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=JSON_VALUE")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["doctor", "prepare", "tokenize", "pilot", "benchmark", "report", "run"]:
        commands.add_parser(name)
    train_parser = commands.add_parser("train")
    train_parser.add_argument("model", choices=["ridge", "text", "relation"])
    train_parser.add_argument("--name")
    train_parser.add_argument("--resume", action="store_true")
    evaluate_parser = commands.add_parser("evaluate")
    evaluate_parser.add_argument("--models", nargs="+", default=["mean", "ridge", "text", "relation"])
    predict_parser = commands.add_parser("predict")
    predict_parser.add_argument("--output")
    fixture_parser = commands.add_parser("fixture")
    fixture_parser.add_argument("--output", default="runs/smoke")
    args = parser.parse_args()
    if args.command == "fixture":
        fixture(args.output)
        return
    cfg = load_config(args.config, args.set)
    # Explicit Python executable for Spark workers, including Windows smoke tests.
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    Path(cfg["run_dir"]).mkdir(parents=True, exist_ok=True)
    manifest_path = Path(cfg["run_dir"]) / "manifest.json"
    if manifest_path.exists() and args.command not in ["prepare", "doctor", "run"]:
        prepared = read_json(manifest_path)["config"]
        for section in ["features", "split"]:
            if cfg[section] != prepared[section]:
                raise ValueError(f"Frozen preprocessing configuration changed: {section}; use its original config or prepare a new run")
    if args.command == "doctor":
        doctor(cfg)
    elif args.command == "prepare":
        from task2.spark_features import prepare
        prepare(cfg)
    elif args.command == "tokenize":
        from task2.data import cache_tokens
        cache_tokens(cfg)
    elif args.command == "pilot":
        from task2.training import pilot
        pilot(cfg)
    elif args.command == "train":
        if args.model == "ridge":
            from task2.baseline import train
            train(cfg, args.name or "ridge")
        else:
            from task2.training import train
            train(cfg, args.model, args.name, args.resume)
    elif args.command == "evaluate":
        from task2.ensemble import evaluate
        evaluate(cfg, args.models)
    elif args.command == "predict":
        predict_pipeline(cfg, args.output)
    elif args.command == "benchmark":
        benchmark(cfg)
    elif args.command == "report":
        report(cfg)
    elif args.command == "run":
        from task2.spark_features import prepare
        from task2 import baseline, ensemble, training
        doctor(cfg)
        if cfg["text"]["mode"] != "scratch" and not cfg["text"]["pretrained_allowed"]:
            raise ValueError("Check competition rules first and set text.pretrained_allowed=true, or use text.mode=scratch")
        with timed(cfg, "total_offline", includes="raw CSV preprocessing, tokenization, training, calibration and component predictions"):
            prepare(cfg)
            baseline.train(cfg)
            training.train(cfg, "relation")
            if cfg["pilot"]["enabled"]:
                cfg = training.pilot(cfg)
            training.train(cfg, "text")
            ensemble.evaluate(cfg, ["mean", "ridge", "text", "relation"])
        predict_pipeline(cfg)
        report(cfg)


if __name__ == "__main__":
    main()
