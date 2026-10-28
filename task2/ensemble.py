"""ID-aligned convex blending, one-time validation audit, and strict CSV output."""
from pathlib import Path
import csv
import math

import numpy as np

from task2.common import iter_rows, read_json, rmse, write_json


def read_predictions(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    result = {r["id"]: float(r["prediction"]) for r in rows}
    if len(result) != len(rows) or not all(math.isfinite(x) for x in result.values()):
        raise ValueError(f"Duplicate ids or non-finite predictions: {path}")
    return result


def aligned_matrix(cfg, names, split, ids):
    columns = []
    expected = set(ids)
    for name in names:
        predictions = read_predictions(Path(cfg["run_dir"]) / "predictions" / name / f"{split}.csv")
        if set(predictions) != expected:
            raise ValueError(f"Prediction IDs do not exactly match {split}: {name}")
        columns.append([predictions[i] for i in ids])
    return np.asarray(columns, dtype=float).T


def project_simplex(values):
    """Euclidean projection onto {w >= 0, sum(w) = 1}."""
    ordered = np.sort(values)[::-1]
    cumulative = np.cumsum(ordered) - 1
    indices = np.arange(1, len(values) + 1)
    rho = np.flatnonzero(ordered - cumulative / indices > 0)[-1]
    return np.maximum(values - cumulative[rho] / (rho + 1), 0.)


def fit_weights(matrix, truth):
    matrix, truth = np.asarray(matrix, dtype=float), np.asarray(truth, dtype=float)
    if matrix.ndim != 2 or len(matrix) != len(truth) or not np.isfinite(matrix).all():
        raise ValueError("Invalid fusion inputs")
    # Small Gram matrix, not N-by-N: typically only 3-4 component models.
    gram, target = matrix.T @ matrix / len(truth), matrix.T @ truth / len(truth)
    rate = 1. / max(float(np.linalg.eigvalsh(gram).max()), 1e-12)
    weights = np.ones(matrix.shape[1]) / matrix.shape[1]
    for _ in range(10000):
        updated = project_simplex(weights - rate * (gram @ weights - target))
        if np.max(np.abs(updated - weights)) < 1e-10:
            weights = updated
            break
        weights = updated
    return weights


def evaluate(cfg, names):
    root = Path(cfg["run_dir"])
    data, metrics = {}, {}
    for split in ["calibration", "validation"]:
        rows = list(iter_rows(root / "features" / split, ["id", "rating", "text_seen", "user_history_count", "product_history_count", "parent_history_count"]))
        ids = [r["id"] for r in rows]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate truth IDs")
        truth = np.array([r["rating"] for r in rows])
        matrix = aligned_matrix(cfg, names, split, ids)
        data[split] = (rows, truth, matrix)
        metrics[split] = {name: rmse(truth, matrix[:, i]) for i, name in enumerate(names)}
    weights = fit_weights(data["calibration"][2], data["calibration"][1])
    rows, truth, matrix = data["validation"]
    fused = matrix @ weights
    fused_score = rmse(truth, fused)
    # One final acceptance comparison, not a source for repeated hyperparameter tuning.
    best_single = min(range(len(names)), key=lambda i: metrics["validation"][names[i]])
    selected = weights if fused_score < metrics["validation"][names[best_single]] else np.eye(len(names))[best_single]
    metrics["calibration"]["blend"] = rmse(data["calibration"][1], data["calibration"][2] @ weights)
    metrics["validation"]["blend"] = fused_score
    metrics["validation"]["selected"] = rmse(truth, matrix @ selected)
    strata = {}
    for label, mask in {
        "text_seen": np.array([r["text_seen"] == 1 for r in rows]),
        "text_unseen": np.array([r["text_seen"] == 0 for r in rows]),
        **{f"{entity}_{bucket}": np.array([low <= r[entity + "_history_count"] < high for r in rows])
           for entity in ["user", "product", "parent"] for bucket, low, high in [("cold", 0, 1), ("1_to_4", 1, 5), ("5_plus", 5, math.inf)]}
    }.items():
        if mask.any():
            strata[label] = {"rows": int(mask.sum()), "rmse": rmse(truth[mask], (matrix @ selected)[mask])}
    result = {"models": names, "calibration_weights": dict(zip(names, weights.tolist())),
              "selected_weights": dict(zip(names, selected.tolist())), "metrics": metrics, "validation_strata": strata,
              "selection_note": "Calibration fits weights; validation performs a single blend-vs-single acceptance comparison."}
    write_json(root / "evaluation.json", result)
    with (root / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["split", "model", "rmse"])
        for split, values in metrics.items():
            writer.writerows((split, name, value) for name, value in values.items())
    print(result)
    return result


def submission(cfg, output=None):
    root = Path(cfg["run_dir"])
    evaluation = read_json(root / "evaluation.json")
    weights = evaluation["selected_weights"]
    names = [n for n, w in weights.items() if w > 0]
    with (Path(cfg["raw_dir"]) / "sampleSubmission.csv").open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["id", "rating"]:
            raise ValueError("Unexpected sample submission schema")
        sample = list(reader)
    ids = [r["id"] for r in sample]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate sample IDs")
    expected = cfg.get("expected_test_rows")
    if expected is not None and len(ids) != expected:
        raise ValueError("Submission row count does not match configuration")
    matrix = aligned_matrix(cfg, names, "test", ids)
    predictions = np.clip(matrix @ np.array([weights[n] for n in names]), 1., 5.)
    if not np.isfinite(predictions).all():
        raise ValueError("Non-finite submission")
    destination = Path(output) if output else root / "submission.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "rating"])
        writer.writerows((ident, format(value, ".10f")) for ident, value in zip(ids, predictions))
    return destination
