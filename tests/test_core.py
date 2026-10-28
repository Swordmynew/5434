import copy
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from task2.common import clean_text, largest_remainder, load_config, rmse, write_json
from task2.ensemble import aligned_matrix, fit_weights, project_simplex, submission


def test_clean_preserves_sentiment():
    assert clean_text('Not &quot;good&quot;!<br />Never again.\n') == 'Not "good"! Never again.'
    assert clean_text(None) == ""


def test_exact_stratified_quotas():
    counts = {1: 1, 2: 7, 3: 13, 4: 29, 5: 50}
    quota = largest_remainder(counts, 17)
    assert sum(quota.values()) == 17
    assert quota == largest_remainder(counts, 17)
    assert all(0 <= quota[k] <= counts[k] for k in counts)
    with pytest.raises(ValueError):
        largest_remainder(counts, 101)


def test_simplex_fit_chooses_useful_model():
    truth = np.array([1., 2., 4., 5.])
    predictions = np.column_stack([truth, 6-truth, np.full(4, 3.)])
    weights = fit_weights(predictions, truth)
    assert weights.sum() == pytest.approx(1.)
    assert weights.min() >= 0
    assert rmse(truth, predictions @ weights) < 1e-5
    assert np.allclose(project_simplex(np.array([-5., 2.])), [0., 1.])


def test_submission_aligns_ids_and_preserves_decimals(tmp_path):
    raw, run = tmp_path / "raw", tmp_path / "run"
    raw.mkdir()
    (run / "predictions" / "a").mkdir(parents=True)
    (raw / "sampleSubmission.csv").write_text("id,rating\n002,3\n001,3\n", encoding="utf-8")
    (run / "predictions" / "a" / "test.csv").write_text("id,prediction\n001,6\n002,2.125\n", encoding="utf-8")
    write_json(run / "evaluation.json", {"selected_weights": {"a": 1.}})
    output = submission({"raw_dir": str(raw), "run_dir": str(run), "expected_test_rows": 2})
    with output.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [r["id"] for r in rows] == ["002", "001"]
    assert [float(r["rating"]) for r in rows] == [2.125, 5.]
    (run / "predictions" / "a" / "test.csv").write_text("id,prediction\n001,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="IDs"):
        submission({"raw_dir": str(raw), "run_dir": str(run), "expected_test_rows": 2})


def test_nonfinite_predictions_rejected(tmp_path):
    from task2.ensemble import read_predictions
    path = tmp_path / "bad.csv"
    path.write_text("id,prediction\na,nan\n")
    with pytest.raises(ValueError):
        read_predictions(path)


def test_config_rejects_unknown_override():
    with pytest.raises(ValueError, match="Unknown"):
        load_config("configs/server.json", ["text.typo=1"])
