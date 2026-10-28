"""One hypothesis test: uniform-label-mixture inversion using weak title hints.

The title hints are NOT verified clean labels. This candidate is experimental,
does not improve calibration RMSE on the observed labels, and assumes the test
target is closer to the pre-mixture rating. No leaderboard score is used to fit
the parameter; training labels and existing candidates remain unchanged.
"""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path


def fit_mixture(diagnostics):
    if diagnostics.get("split") != "calibration":
        raise ValueError("Use the calibration-only diagnostic file")
    groups = diagnostics["rating_title_hints"]
    total, matching = 0, 0
    for rating in range(1, 6):
        group = groups[str(rating)]
        counts = group["observed_label_counts"]
        if any(int(counts[str(k)]) != counts[str(k)] or counts[str(k)] < 0 for k in range(1, 6)):
            raise ValueError("Invalid title-hint label counts")
        n = sum(counts[str(k)] for k in range(1, 6))
        if n != group["rows"]:
            raise ValueError("Title-hint row count mismatch")
        total += n
        matching += counts[str(rating)]
    if not total or total > diagnostics["label_count"]:
        raise ValueError("Invalid total title-hint count")
    rho = (1 - matching / total) * 5 / 4
    if not 0 <= rho < 1 - 1e-6:
        raise ValueError("Non-identifiable or inadmissible uniform mixture")
    return rho, total, matching


def correct(value, rho):
    return min(5., max(1., (value - 3*rho) / (1-rho)))


def rmse(y, prediction):
    return math.sqrt(math.fsum((a-b)**2 for a, b in zip(y, prediction))/len(y))


def submission_bytes(ids, values):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(["id", "rating"])
    writer.writerows((i, format(v, ".10f")) for i, v in zip(ids, values))
    return stream.getvalue().encode("utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--postcheck-dir", required=True)
    parser.add_argument("--predictions", default="runs/main/predictions/text/test.csv")
    parser.add_argument("--sample", default="data/raw/sampleSubmission.csv")
    parser.add_argument("--output", default="runs/main/submission_text_noise_hypothesis.csv")
    parser.add_argument("--expected-rows", type=int, default=10000)
    args = parser.parse_args()
    source = Path(args.postcheck_dir)
    diagnostic_path = source / "diagnostics.json"
    diagnostics = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    rho, n, matching = fit_mixture(diagnostics)
    with (source / "calibration_predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
        calibration = list(csv.DictReader(stream))
    if len(calibration) != diagnostics["label_count"] or len({r["id"] for r in calibration}) != len(calibration):
        raise ValueError("Invalid calibration rows/IDs")
    y = [float(r["rating"]) for r in calibration]
    raw_calibration = [float(r["pred_text"]) for r in calibration]
    if not all(math.isfinite(v) for v in y + raw_calibration) or any(v not in [1, 2, 3, 4, 5] for v in y):
        raise ValueError("Invalid calibration values")
    original_rmse = rmse(y, raw_calibration)
    if not math.isclose(original_rmse, diagnostics["models"]["text"]["rmse"], abs_tol=1e-9, rel_tol=0):
        raise ValueError("Calibration data does not reproduce the diagnostic RMSE")
    with Path(args.sample).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["id", "rating"]:
            raise ValueError("Unexpected sample columns")
        ids = [r["id"] for r in reader]
    with Path(args.predictions).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["id", "prediction"]:
            raise ValueError("Unexpected component prediction columns")
        rows = list(reader)
    lookup = {r["id"]: float(r["prediction"]) for r in rows}
    if len(ids) != args.expected_rows or len(set(ids)) != len(ids) or len(lookup) != len(rows) or set(lookup) != set(ids):
        raise ValueError("Prediction/sample IDs or row counts do not match")
    raw = [lookup[i] for i in ids]
    if not all(math.isfinite(v) and 1 <= v <= 5 for v in raw):
        raise ValueError("Invalid saved text predictions")
    corrected = [correct(v, rho) for v in raw]
    before_clipping = [(v-3*rho)/(1-rho) for v in raw]
    data = submission_bytes(ids, corrected)
    destination = Path(args.output)
    metadata_path = destination.with_suffix(".json")
    if destination.exists() or metadata_path.exists():
        raise FileExistsError("Use a new output path; existing candidate/metadata are preserved")
    metadata = {
        "experiment": "uniform_label_mixture_hypothesis",
        "parameter_fit_source": "calibration rating-title hint counts only; no public score input",
        "weak_title_hints_are_verified_clean_labels": False,
        "title_hint_rows": n, "matching_labels": matching, "mismatching_labels": n-matching,
        "observed_hint_disagreement_fraction": (n-matching)/n,
        "uniform_replacement_mixture_rho": rho,
        "assumptions": [
            "Rating titles are informative weak proxies for a pre-mixture rating, not verified truth.",
            "Observed labels follow (1-rho)*pre-mixture distribution + rho*Uniform(1,2,3,4,5), independently of features.",
            "The test target is closer to the pre-mixture rating scale. This is unverified.",
            "The text model approximates the observed conditional mean sufficiently well for inversion."],
        "formula": "clip((text_prediction - 3*rho)/(1-rho),1,5)",
        "transition_diagonal_under_assumption": 1-4*rho/5,
        "transition_off_diagonal_under_assumption": rho/5,
        "observed_calibration_rmse_before": original_rmse,
        "observed_calibration_rmse_after": rmse(y, [correct(v, rho) for v in raw_calibration]),
        "selection_note": "Experimental target-scale hypothesis; not selected by a lower observed-label calibration RMSE.",
        "test_rows": len(ids), "test_minimum": min(corrected), "test_maximum": max(corrected),
        "test_mean": math.fsum(corrected)/len(corrected),
        "test_clipped_low_rows": sum(v<1 for v in before_clipping),
        "test_clipped_high_rows": sum(v>5 for v in before_clipping),
        "diagnostics_sha256": hashlib.sha256(diagnostic_path.read_bytes()).hexdigest(),
        "source_prediction_sha256": hashlib.sha256(Path(args.predictions).read_bytes()).hexdigest(),
        "baseline_text_submission_sha256": hashlib.sha256(submission_bytes(ids, raw)).hexdigest(),
        "candidate_sha256": hashlib.sha256(data).hexdigest(),
        "public_score": None,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".csv.tmp")
    temporary.write_bytes(data)
    temporary.replace(destination)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print("HYPOTHESIS CANDIDATE:", destination.resolve())
    print("rho estimated from weak title hints:", rho)
    print("Observed calibration RMSE before/after:", original_rmse, metadata["observed_calibration_rmse_after"])
    print("SHA256:", metadata["candidate_sha256"])
    print("Unverified target-scale hypothesis; keep the original text-only submission.")


if __name__ == "__main__":
    main()
