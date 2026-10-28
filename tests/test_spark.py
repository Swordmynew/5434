"""Real Spark integration checks; run with Java 17 and writable local scratch."""
import os
from pathlib import Path
import sys

import pytest

from task2.common import load_config, read_json


@pytest.mark.spark
def test_full_etl_and_crossfit_excludes_entire_fold(tmp_path):
    from pyspark.sql import functions as F
    from task2.cli import fixture
    from task2.spark_features import prepare, session, enrich
    fixture(tmp_path)
    cfg = load_config(tmp_path / "config.json")
    os.environ["PYSPARK_PYTHON"] = sys.executable
    prepare(cfg)
    manifest = read_json(Path(cfg["run_dir"]) / "manifest.json")
    assert manifest["counts"] == {"train": 500, "calibration": 50, "validation": 50, "test": 20}
    from task2 import baseline, ensemble
    from task2.cli import predict_pipeline
    baseline.train(cfg)
    ensemble.evaluate(cfg, ["mean", "ridge"])
    original_submission = ensemble.submission(cfg).read_bytes()
    assert predict_pipeline(cfg).read_bytes() == original_submission
    spark = session(cfg)
    try:
        train = spark.read.parquet(str(Path(cfg["run_dir"]) / "clean")).filter("split='train'")
        features = spark.read.parquet(str(Path(cfg["run_dir"]) / "features" / "train"))
        chosen = train.first()
        row = features.filter(F.col("id") == chosen["id"]).first()
        complement = train.filter(F.col("fold") != chosen["fold"])
        entity = complement.filter(F.col("user_id") == chosen["user_id"]).agg(F.count("rating"), F.sum("rating")).first()
        prior = complement.agg(F.avg("rating")).first()[0]
        expected = ((entity[1] or 0.) + 20.*prior)/(entity[0]+20.)
        assert row["user_mean_value"]*2+3 == pytest.approx(expected)
        # Changing EVERY label in the held-out fold cannot change that fold's features.
        modified = train.withColumn("rating", F.when(F.col("fold") == chosen["fold"], F.lit(1.)).otherwise(F.col("rating")))
        from task2.spark_features import build_history
        build_history(spark, modified, cfg)
        changed = enrich(spark, modified.filter(F.col("id") == chosen["id"]), cfg, crossfit=True).first()
        assert changed["numeric"] == pytest.approx(row["numeric"])
        test = spark.read.parquet(str(Path(cfg["run_dir"]) / "features" / "test")).filter("id='19'").first()
        assert test["user_idx"] == test["product_idx"] == test["parent_idx"] == 0
        assert test["user_history_count"] == 0
    finally:
        spark.stop()
