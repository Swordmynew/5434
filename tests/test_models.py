from types import SimpleNamespace

import pytest
import torch
from torch import nn

from task2.common import NUMERIC_NAMES
from task2.data import Collate, ShardedRows, pack_pair, truncate_head_tail
from task2.models import RelationRatingModel, TextRatingModel


class TinyEncoder(nn.Module):
    config = SimpleNamespace(hidden_size=8)

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(20, 8)

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def test_expected_rating_loss_and_missing_body():
    model = TextRatingModel(TinyEncoder(), dropout=0.)
    ids = torch.tensor([[1, 2, 3], [1, 4, 0]])
    prediction, logits = model(ids, ids.ne(0).long(), torch.zeros_like(ids), torch.zeros_like(ids))
    assert torch.isfinite(prediction).all()
    assert ((prediction >= 1) & (prediction <= 5)).all()
    loss = model.loss(prediction, logits, torch.tensor([1., 5.]), 0.2)
    loss.backward()
    assert model.head[-1].weight.grad.abs().sum() > 0


def test_deberta_v3_compatible_encoder_interface():
    from transformers import DebertaV2Config, DebertaV2Model
    encoder = DebertaV2Model(DebertaV2Config(vocab_size=20, hidden_size=8, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=16, max_position_embeddings=32,
        type_vocab_size=0, relative_attention=True, pos_att_type=["p2c", "c2p"]))
    model = TextRatingModel(encoder, dropout=0.)
    ids = torch.tensor([[1, 3, 2, 5, 2]])
    prediction, logits = model(ids, torch.ones_like(ids), torch.zeros_like(ids), torch.tensor([[0, 0, 0, 1, 0]]))
    model.loss(prediction, logits, torch.tensor([4.]), 0.2).backward()
    assert torch.isfinite(prediction).all()


def test_relation_unknown_entities_and_ablation():
    model = RelationRatingModel([3, 4, 2], len(NUMERIC_NAMES), 3.5, dimension=4, drop_user=True, drop_parent=True)
    model.eval()
    numeric = torch.randn(2, len(NUMERIC_NAMES))
    entities = torch.tensor([[0, 0, 0], [1, 2, 1]])
    prediction = model(entities, numeric)
    changed = numeric.clone()
    for i, name in enumerate(NUMERIC_NAMES):
        if name.startswith("user_") or name.startswith("parent_"):
            changed[:, i] = 100.
    alternative = entities.clone()
    alternative[:, 0], alternative[:, 2] = 2, 0
    assert torch.allclose(prediction, model(alternative, changed))
    assert torch.isfinite(prediction).all()
    assert all(torch.all(e.weight[0] == 0) for e in model.embeddings)


def test_head_tail_and_title_priority(tmp_path):
    from transformers import BertTokenizer
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "title", "start", "end", "middle"]
    path = tmp_path / "vocab.txt"
    path.write_text("\n".join(vocab))
    tokenizer = BertTokenizer(vocab_file=str(path))
    packed = pack_pair(tokenizer, "title title", "start " + "middle " * 20 + "end",
                       {"max_length": 10, "title_tokens": 2, "drop_title": False})
    assert len(packed["input_ids"]) == 10
    assert packed["input_ids"][1:3] == [5, 5]
    assert packed["input_ids"][-2] == 7
    assert sum(packed["body_mask"]) == 5
    assert truncate_head_tail([1, 2, 3, 4], 1) == [1]


def test_streaming_shards_do_not_drop_rows(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    for i in range(3):
        pq.write_table(pa.Table.from_pylist([{"id": str(i*7+j), "rating": 3.} for j in range(7)]), tmp_path / f"{i}.parquet")
    first = [r["id"] for r in ShardedRows(tmp_path, seed=5, shuffle=True)]
    second = [r["id"] for r in ShardedRows(tmp_path, seed=5, shuffle=True)]
    assert first == second
    assert len(first) == len(set(first)) == 21


def test_training_reload_and_resume(tmp_path):
    import json
    import pyarrow as pa
    import pyarrow.parquet as pq
    from task2.common import load_config, write_json
    from task2.training import train, predict
    cfg = load_config("configs/server.json")
    cfg.update(run_dir=str(tmp_path), device="cpu")
    cfg["relation"].update(batch_size=4, workers=0, epochs=1, max_hours=0.05)
    rows = [{"id": str(i), "rating": float(i%5+1), "user_idx": i%2+1, "product_idx": i%3+1,
             "parent_idx": 1, "numeric": [0.] * len(NUMERIC_NAMES)} for i in range(20)]
    for split in ["train", "calibration", "validation", "test"]:
        path = tmp_path / "features" / split
        path.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(rows), path / "data.parquet")
    write_json(tmp_path / "manifest.json", {"counts": {"train": 20}, "cardinalities": [3, 4, 2], "numeric_names": NUMERIC_NAMES, "mean": 3.})
    train(cfg, "relation")
    before = (tmp_path / "predictions" / "relation" / "test.csv").read_bytes()
    predict(cfg, "relation")
    assert (tmp_path / "predictions" / "relation" / "test.csv").read_bytes() == before
    cfg["relation"]["epochs"] = 2
    train(cfg, "relation", resume=True)
    assert (tmp_path / "models" / "relation" / "epoch_2.json").exists()
