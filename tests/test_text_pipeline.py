"""CPU text training, frozen inference, fusion, and output without external weights."""
from pathlib import Path
import csv

import pyarrow as pa
import pyarrow.parquet as pq

from task2.common import NUMERIC_NAMES, load_config, write_json


def test_scratch_text_training_and_reloaded_submission(tmp_path):
    from task2.data import cache_tokens
    from task2.training import train, predict
    from task2.ensemble import evaluate, submission
    cfg = load_config("configs/server.json")
    raw = tmp_path / "raw"
    raw.mkdir()
    cfg.update(run_dir=str(tmp_path), raw_dir=str(raw), device="cpu", expected_test_rows=10)
    cfg["text"].update(mode="scratch", scratch_vocab_size=100, scratch_hidden=16, scratch_layers=1,
                         max_length=24, title_tokens=6, batch_size=4, accumulation=3, epochs=1,
                         workers=0, tokenizer_workers=1, amp=False, gradient_checkpointing=False,
                         evaluate_every_steps=0, max_hours=0.05)
    records = [{"id": str(i), "rating": float(i%5+1), "title": "good" if i%2 else "bad",
                "comment": "" if i==0 else "works well and useful", "text_seen": 0,
                "user_history_count": 1., "product_history_count": 2., "parent_history_count": 3.}
                for i in range(10)]
    for split in ["train", "calibration", "validation", "test"]:
        path = tmp_path / "features" / split
        path.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(records), path / "data.parquet")
    write_json(tmp_path / "manifest.json", {"counts": {"train": 10}, "cardinalities": [2, 2, 2], "numeric_names": NUMERIC_NAMES, "mean": 3.})
    (raw / "sampleSubmission.csv").write_text("id,rating\n" + "".join(f"{i},3\n" for i in reversed(range(10))))
    train(cfg, "text")
    evaluate(cfg, ["text"])
    original = submission(cfg).read_bytes()
    # Simulate a Spark rewrite with completely different test shard names.
    old = tmp_path / "features" / "test" / "data.parquet"
    old.rename(old.with_name("new-part-name.parquet"))
    predict(cfg, "text")
    assert len(list((tmp_path / "tokens" / "test").glob("*.parquet"))) == 1
    assert submission(cfg).read_bytes() == original
