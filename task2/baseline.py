"""Distributed TF-IDF + Ridge comparison; main models are custom PyTorch code."""
from pathlib import Path

from task2.common import read_json, timed, write_json, write_predictions
from task2.spark_features import session


def documents(frame, drop_title=False):
    from pyspark.sql import functions as F
    # Namespace title/body and unigrams/bigrams to keep fields distinguishable.
    title = F.split(F.trim(F.regexp_replace(F.lower("title"), r"[^\w']+", " ")), r"\s+")
    body = F.split(F.trim(F.regexp_replace(F.lower("comment"), r"[^\w']+", " ")), r"\s+")
    title = F.filter(title, lambda x: F.length(x) > 0)
    body = F.filter(body, lambda x: F.length(x) > 0)
    if drop_title:
        title = F.array().cast("array<string>")
    frame = frame.withColumn("title_words", title).withColumn("body_words", body)
    for field in ["title", "body"]:
        words = F.col(field + "_words")
        bigrams = F.when(F.size(words) > 1, F.transform(F.sequence(F.lit(1), F.size(words) - 1),
            lambda i: F.concat(F.element_at(words, i), F.lit("_"), F.element_at(words, i+1)))).otherwise(F.array().cast("array<string>"))
        frame = frame.withColumn(field + "_tokens", F.concat(
            F.transform(words, lambda x: F.concat(F.lit(field + ":u:"), x)),
            F.transform(bigrams, lambda x: F.concat(F.lit(field + ":b:"), x))))
    return frame.withColumn("tokens", F.concat("title_tokens", "body_tokens"))


def _write(frame, destination):
    # Predictions are bounded by calibration/validation/test sizes, not train size.
    write_predictions(destination, ({"id": r["id"], "prediction": r["prediction"]}
                      for r in frame.select("id", "prediction").toLocalIterator()))


def train(cfg, name="ridge"):
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import HashingTF, IDF, Normalizer
    from pyspark.ml.regression import LinearRegression
    root = Path(cfg["run_dir"])
    target = root / "models" / name
    if target.exists():
        raise FileExistsError("Use a new baseline --name to preserve fitted artifacts")
    spark = session(cfg, "task2-ridge")
    try:
        settings = cfg["baseline"]
        with timed(cfg, "training", model=name, kind="ridge"):
            train_frame = documents(spark.read.parquet(str(root / "features" / "train")), settings["drop_title"])
            stages = [HashingTF(inputCol="tokens", outputCol="tf", numFeatures=settings["hash_features"]),
                      IDF(inputCol="tf", outputCol="tfidf"), Normalizer(inputCol="tfidf", outputCol="features"),
                      LinearRegression(featuresCol="features", labelCol="rating", solver="l-bfgs", elasticNetParam=0.,
                                       regParam=settings["reg_param"], maxIter=settings["max_iter"], standardization=False)]
            model = Pipeline(stages=stages).fit(train_frame)
            iterations = model.stages[-1].summary.totalIterations
            model.save(str(target / "pipeline"))
            write_json(target / "settings.json", {"drop_title": settings["drop_title"], "iterations": iterations})
        mean = read_json(root / "manifest.json")["mean"]
        from pyspark.sql import functions as F
        for split in ["calibration", "validation", "test"]:
            frame = spark.read.parquet(str(root / "features" / split))
            _write(frame.withColumn("prediction", F.lit(mean)), root / "predictions" / "mean" / f"{split}.csv")
            with timed(cfg, "model_prediction", model=name, split=split):
                _write(model.transform(documents(frame, settings["drop_title"])), root / "predictions" / name / f"{split}.csv")
    finally:
        spark.stop()


def predict(cfg, name="ridge", spark=None):
    from pyspark.ml import PipelineModel
    own = spark is None
    spark = spark or session(cfg, "task2-ridge-predict")
    root = Path(cfg["run_dir"])
    try:
        model = PipelineModel.load(str(root / "models" / name / "pipeline"))
        settings = read_json(root / "models" / name / "settings.json")
        frame = spark.read.parquet(str(root / "features" / "test"))
        _write(model.transform(documents(frame, settings["drop_title"])), root / "predictions" / name / "test.csv")
    finally:
        if own:
            spark.stop()
