"""Spark ETL, deterministic splits, and out-of-fold rating statistics.

Only small aggregates (five labels/folds) reach the driver. Entity dictionaries,
history tables, joins and text duplication checks remain distributed.
"""
from pathlib import Path
import math
import os
import sys
import time

from task2.common import (BASE_NUMERIC, ENTITY_FIELDS, ENTITY_PREFIXES, MISSING,
                         NUMERIC_NAMES, clean_text, largest_remainder, read_json,
                         stage_raw, timed, write_json)


def session(cfg, name="task2"):
    from pyspark.sql import SparkSession
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    root = Path(cfg["run_dir"]).resolve()
    events = root / "spark-events"
    events.mkdir(parents=True, exist_ok=True)
    s = cfg["spark"]
    builder = (SparkSession.builder.master(s["master"]).appName(name)
               .config("spark.sql.shuffle.partitions", str(s["partitions"]))
               .config("spark.driver.memory", s["driver_memory"])
               .config("spark.executor.memory", s["executor_memory"])
               .config("spark.executor.cores", str(s["executor_cores"]))
               .config("spark.cores.max", str(s["cores_max"]))
               .config("spark.sql.session.timeZone", "UTC")
               .config("spark.sql.parquet.compression.codec", "snappy")
               .config("spark.eventLog.enabled", "true")
               .config("spark.eventLog.dir", events.as_uri()))
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def read_clean(spark, path, labelled):
    from pyspark.sql import functions as F, types as T
    columns = ["id", "title", "user_id", "comment", "prod_id", "parent_prod_id", "time", "votes", "purchased"]
    if labelled:
        columns += ["rating"]
    schema = T.StructType([T.StructField(x, T.StringType()) for x in columns])
    frame = (spark.read.schema(schema).option("header", True).option("enforceSchema", False)
             .option("multiLine", True).option("escape", '"').option("mode", "FAILFAST").csv(str(path)))
    if frame.filter(F.col("id").isNull()).limit(1).count():
        raise ValueError("Missing record id")
    if frame.groupBy("id").count().filter("count > 1").limit(1).count():
        raise ValueError("Duplicate record ids in input")
    if labelled:
        frame = frame.withColumn("rating", F.col("rating").cast("double"))
        if frame.filter(F.col("rating").isNull() | ~F.col("rating").isin([1., 2., 3., 4., 5.])).limit(1).count():
            raise ValueError("Invalid rating; refusing to silently drop records")
    else:
        frame = frame.withColumn("rating", F.lit(None).cast("double"))
    normalize = F.udf(clean_text, T.StringType())
    for field in ["title", "comment"]:
        frame = frame.withColumn(field, normalize(F.col(field)))
        frame = frame.withColumn(field + "_missing", (F.length(field) == 0).cast("double"))
        frame = frame.withColumn(field + "_length_value", F.log1p(F.length(field)) / 10.)
    for field in ENTITY_FIELDS:
        frame = frame.withColumn(field, F.when(F.length(F.trim(F.coalesce(F.col(field), F.lit("")))) == 0, MISSING).otherwise(F.col(field)))
    timestamp = F.to_timestamp(F.from_unixtime(F.col("time").cast("double") / 1000.))
    frame = (frame.withColumn("time_missing", timestamp.isNull().cast("double"))
             .withColumn("year_value", F.coalesce((F.year(timestamp) - 2000.) / 25., F.lit(0.)))
             .withColumn("month_sin", F.coalesce(F.sin(F.month(timestamp) * (2 * math.pi / 12)), F.lit(0.)))
             .withColumn("month_cos", F.coalesce(F.cos(F.month(timestamp) * (2 * math.pi / 12)), F.lit(0.)))
             .withColumn("votes_value", F.log1p(F.greatest(F.coalesce(F.col("votes").cast("double"), F.lit(0.)), F.lit(0.))) / 10.)
             .withColumn("purchased_value", (F.lower(F.coalesce(F.col("purchased"), F.lit(""))) == "true").cast("double"))
             .withColumn("text_key", F.sha2(F.concat(F.lower("title"), F.lit("\x1f"), F.lower("comment")), 256)))
    return frame


def assign_splits(frame, cfg):
    from pyspark.sql import functions as F, Window
    counts = {int(r["rating"]): r["count"] for r in frame.groupBy("rating").count().collect()}
    cal_n, val_n = cfg["split"]["calibration_size"], cfg["split"]["validation_size"]
    if cal_n <= 0 or val_n <= 0 or cal_n + val_n >= sum(counts.values()):
        raise ValueError("Calibration/validation sizes must be positive and leave training rows")
    if cfg["split"]["strategy"] == "group":
        # Approximate sizes, exact text-group exclusion. Never split a group.
        fraction = F.pmod(F.xxhash64("text_key", F.lit(cfg["seed"])), F.lit(1000000)) / 1000000.
        n = sum(counts.values())
        assigned = frame.withColumn("split", F.when(fraction < cal_n / n, "calibration")
                    .when(fraction < (cal_n + val_n) / n, "validation").otherwise("train"))
    elif cfg["split"]["strategy"] == "stratified":
        cal = largest_remainder(counts, cal_n)
        remaining = {k: counts[k] - cal[k] for k in counts}
        val = largest_remainder(remaining, val_n)
        thresholds = frame.sparkSession.createDataFrame([(float(k), cal[k], cal[k]+val[k]) for k in counts], ["rating", "n_cal", "n_val"])
        window = Window.partitionBy("rating").orderBy(F.xxhash64("id", F.lit(cfg["seed"])), "id")
        assigned = (frame.withColumn("rank", F.row_number().over(window)).join(F.broadcast(thresholds), "rating")
                    .withColumn("split", F.when(F.col("rank") <= F.col("n_cal"), "calibration")
                    .when(F.col("rank") <= F.col("n_val"), "validation").otherwise("train"))
                    .drop("rank", "n_cal", "n_val"))
    else:
        raise ValueError("split.strategy must be stratified or group")
    return assigned.withColumn("fold", F.when(F.col("split") == "train",
            F.pmod(F.xxhash64("id", F.lit(cfg["seed"] + 1)), F.lit(cfg["split"]["folds"]))).otherwise(-1).cast("int"))


def stats_exprs():
    from pyspark.sql import functions as F
    return [F.count("rating").cast("double").alias("n"), F.sum("rating").alias("s")] + [
        F.sum(F.when(F.col("rating") == k, 1.).otherwise(0.)).alias(f"c{k}") for k in range(1, 6)]


def build_history(spark, train, cfg):
    from pyspark.sql import functions as F, types as T
    root = Path(cfg["run_dir"]) / "history"
    total = train.agg(*stats_exprs()).first().asDict()
    folds = {r["fold"]: r.asDict() for r in train.groupBy("fold").agg(*stats_exprs()).collect()}
    priors = []
    for fold in [-1, *range(cfg["split"]["folds"])]:
        excluded = folds.get(fold, {})
        n = total["n"] - excluded.get("n", 0.)
        if n <= 0:
            raise ValueError("Empty cross-fitting complement")
        priors.append((fold, (total["s"] - excluded.get("s", 0.)) / n, *[
            (total[f"c{k}"] - excluded.get(f"c{k}", 0.)) / n for k in range(1, 6)]))
    write_json(root / "priors.json", {"rows": priors, "mean": total["s"] / total["n"], "count": total["n"]})
    cardinalities = []
    for field, prefix in zip(ENTITY_FIELDS, ENTITY_PREFIXES):
        valid = train.filter(F.col(field) != MISSING)
        totals = valid.groupBy(field).agg(*stats_exprs())
        totals.write.mode("overwrite").parquet(str(root / prefix / "totals"))
        valid.groupBy(field, "fold").agg(*stats_exprs()).write.mode("overwrite").parquet(str(root / prefix / "folds"))
        # A globally ordered distributed enumeration: no million-label list on driver.
        pairs = (totals.select(field).orderBy(field).rdd.map(lambda r: r[0]).zipWithIndex()
                 .map(lambda pair: (pair[0], pair[1] + 1)))
        mapping = spark.createDataFrame(pairs, T.StructType([T.StructField(field, T.StringType()), T.StructField(prefix + "_idx", T.LongType())]))
        mapping.write.mode("overwrite").parquet(str(root / prefix / "mapping"))
        cardinalities.append(totals.count() + 1)
    train.select("text_key").distinct().write.mode("overwrite").parquet(str(root / "text_keys"))
    return cardinalities


def enrich(spark, frame, cfg, crossfit=False):
    from pyspark.sql import functions as F
    root = Path(cfg["run_dir"]) / "history"
    prior_data = read_json(root / "priors.json")
    prior_cols = ["fold", "prior_mean", *[f"prior_{k}" for k in range(1, 6)]]
    priors = spark.createDataFrame(prior_data["rows"], prior_cols)
    if not crossfit:
        frame = frame.withColumn("fold", F.lit(-1))
    frame = frame.join(F.broadcast(priors), "fold", "left")
    alpha = cfg["features"]["smoothing"]
    for field, prefix in zip(ENTITY_FIELDS, ENTITY_PREFIXES):
        total = spark.read.parquet(str(root / prefix / "totals"))
        frame = frame.join(total.select(field, *[F.col(x).alias("t_" + x) for x in ["n", "s", "c1", "c2", "c3", "c4", "c5"]]), field, "left")
        if crossfit:
            folds = spark.read.parquet(str(root / prefix / "folds"))
            frame = frame.join(folds.select(field, "fold", *[F.col(x).alias("f_" + x) for x in ["n", "s", "c1", "c2", "c3", "c4", "c5"]]), [field, "fold"], "left")
        else:
            for name in ["n", "s", "c1", "c2", "c3", "c4", "c5"]:
                frame = frame.withColumn("f_" + name, F.lit(0.))
        for name in ["n", "s", "c1", "c2", "c3", "c4", "c5"]:
            frame = frame.withColumn("h_" + name, F.coalesce(F.col("t_"+name), F.lit(0.)) - F.coalesce(F.col("f_"+name), F.lit(0.)))
        frame = (frame.withColumn(prefix + "_history_count", F.col("h_n"))
                 .withColumn(prefix + "_count_value", F.log1p("h_n") / 10.)
                 .withColumn(prefix + "_mean_value", ((F.col("h_s") + alpha * F.col("prior_mean")) / (F.col("h_n") + alpha) - 3.) / 2.))
        for k in range(1, 6):
            frame = frame.withColumn(f"{prefix}_prob_{k}", (F.col(f"h_c{k}") + alpha * F.col(f"prior_{k}")) / (F.col("h_n") + alpha))
        frame = frame.drop(*[p+x for p in ["t_", "f_", "h_"] for x in ["n", "s", "c1", "c2", "c3", "c4", "c5"]])
        frame = (frame.join(spark.read.parquet(str(root / prefix / "mapping")), field, "left")
                 .fillna({prefix + "_idx": 0}))
    keys = spark.read.parquet(str(root / "text_keys")).withColumn("text_seen", F.lit(1))
    return (frame.join(keys, "text_key", "left").fillna({"text_seen": 0})
            .withColumn("numeric", F.array(*[F.col(x).cast("float") for x in NUMERIC_NAMES])))


def prepare(cfg):
    root = Path(cfg["run_dir"])
    if (root / "manifest.json").exists():
        raise FileExistsError("Run already prepared; use a new run_dir to protect splits and histories")
    raw = stage_raw(cfg)
    spark = session(cfg, "task2-prepare")
    try:
        with timed(cfg, "preprocessing"):
            assigned = assign_splits(read_clean(spark, raw / "train.csv", True), cfg)
            assigned.repartition(cfg["spark"]["partitions"]).write.partitionBy("split").mode("overwrite").parquet(str(root / "clean"))
            all_rows = spark.read.parquet(str(root / "clean"))
            counts = {r["split"]: r["count"] for r in all_rows.groupBy("split").count().collect()}
            if any(counts.get(s, 0) == 0 for s in ["train", "calibration", "validation"]):
                raise ValueError("Empty split; increase fixture or change split sizes")
            train = all_rows.filter("split = 'train'")
            cardinalities = build_history(spark, train, cfg)
            for split in ["train", "calibration", "validation"]:
                features = enrich(spark, all_rows.filter(F_sql_split(split)), cfg, crossfit=split == "train")
                features.repartition(cfg["spark"]["partitions"]).write.mode("overwrite").parquet(str(root / "features" / split))
            test_count = prepare_test(cfg, spark)
            prior = read_json(root / "history" / "priors.json")
            write_json(root / "manifest.json", {"counts": {**counts, "test": test_count}, "mean": prior["mean"],
                "cardinalities": cardinalities, "numeric_names": NUMERIC_NAMES, "config": cfg,
                "raw_files": {p.name: {"bytes": p.stat().st_size} for p in raw.glob("*.csv")}})
    finally:
        spark.stop()


def F_sql_split(split):
    from pyspark.sql import functions as F
    return F.col("split") == split


def prepare_test(cfg, spark=None):
    own = spark is None
    spark = spark or session(cfg, "task2-test-features")
    try:
        test = read_clean(spark, Path(cfg["raw_dir"]) / "test.csv", False)
        count = test.count()
        expected = cfg.get("expected_test_rows")
        if expected is not None and count != expected:
            raise ValueError(f"Expected {expected} test rows, got {count}")
        enriched = enrich(spark, test, cfg)
        # Stable shard membership/order keeps dynamic-padding inference batches
        # identical after raw-test regeneration, including on the GPU.
        from pyspark.sql import functions as F
        (enriched.repartition(min(8, cfg["spark"]["partitions"]), F.col("id")).sortWithinPartitions("id")
         .write.mode("overwrite").parquet(str(Path(cfg["run_dir"]) / "features" / "test")))
        return count
    finally:
        if own:
            spark.stop()
