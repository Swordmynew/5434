"""Explicit PyTorch optimization, calibration-only early stopping, and resume."""
from __future__ import annotations

import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from task2.common import (device_for, read_json, rmse, seed_everything, synchronize,
                         timed, write_json, write_predictions)
from task2.data import cache_tokens, loader, tokenizer_setup
from task2.models import RelationRatingModel, TextRatingModel


def build_model(cfg, kind, model_dir, initialize=True):
    manifest = read_json(Path(cfg["run_dir"]) / "manifest.json")
    if kind == "relation":
        s = cfg["relation"]
        return RelationRatingModel(manifest["cardinalities"], len(manifest["numeric_names"]), manifest["mean"],
                                   s["embedding_dim"], s["drop_user"], s["drop_parent"]), 0
    from transformers import AutoConfig, AutoModel, AutoTokenizer, BertConfig, BertModel
    settings = cfg["text"]
    tokenizer = AutoTokenizer.from_pretrained(Path(cfg["run_dir"]) / "tokenizer", local_files_only=True, use_fast=False)
    config_path = model_dir / "encoder_config"
    if not initialize:
        encoder = AutoModel.from_config(AutoConfig.from_pretrained(config_path, local_files_only=True))
    elif settings["mode"] == "scratch":
        encoder = BertModel(BertConfig(vocab_size=len(tokenizer), hidden_size=settings["scratch_hidden"],
                           num_hidden_layers=settings["scratch_layers"], num_attention_heads=4,
                           intermediate_size=settings["scratch_hidden"] * 4, max_position_embeddings=settings["max_length"]+8))
    else:
        if not settings["pretrained_allowed"]:
            raise ValueError("Pretrained weights not authorized in config")
        encoder = AutoModel.from_pretrained(settings["backbone"], revision=settings["revision"])
    if initialize:
        encoder.config.save_pretrained(config_path)
        write_json(model_dir / "backbone_source.json", {"backbone": settings["backbone"], "revision": settings["revision"],
                   "resolved_commit": getattr(encoder.config, "_commit_hash", None), "mode": settings["mode"]})
    if settings["gradient_checkpointing"]:
        encoder.gradient_checkpointing_enable()
    return TextRatingModel(encoder, settings["dropout"]), tokenizer.pad_token_id


def forward(model, kind, batch, device):
    if kind == "text":
        return model(**{k: batch[k].to(device, non_blocking=True) for k in ["input_ids", "attention_mask", "token_type_ids", "body_mask"]})
    return model(batch["entities"].to(device, non_blocking=True), batch["numeric"].to(device, non_blocking=True)), None


@torch.no_grad()
def predict_loader(model, kind, batches, device, amp=False):
    model.eval()
    records, truth = [], []
    for batch in batches:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            predictions, _ = forward(model, kind, batch, device)
        records.extend({"id": ident, "prediction": float(value)} for ident, value in zip(batch["id"], predictions.float().clamp(1, 5).cpu().tolist()))
        truth.extend(batch["rating"].tolist())
    if not records:
        raise ValueError("No prediction rows; check Parquet shards and worker count")
    return records, truth


def _atomic_torch_save(value, path):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def train(cfg, kind, name=None, resume=False, emit_predictions=True):
    name = name or kind
    destination = Path(cfg["run_dir"]) / "models" / name
    if (destination / "best.pt").exists() and not resume:
        raise FileExistsError("Model already exists; select a new --name or --resume")
    destination.mkdir(parents=True, exist_ok=True)
    if kind == "text":
        cache_tokens(cfg)
    if device_for(cfg).type == "cuda":
        import json
        timing_path = Path(cfg["run_dir"]) / "timings.jsonl"
        timings = [json.loads(line) for line in timing_path.read_text().splitlines()] if timing_path.exists() else []
        used = sum(r["seconds"] for r in timings if r["stage"] == "training" and r.get("device") == "cuda") / 3600
        available = cfg["gpu_budget_hours"] - 8 - used
        if available <= 0:
            raise RuntimeError("GPU training budget exhausted; eight hours reserved for final inference/reproduction")
        cfg[kind]["max_hours"] = min(cfg[kind]["max_hours"], available)
    # CUDA OOM retries are explicit; effective text batch size remains unchanged.
    while True:
        try:
            with timed(cfg, "training", model=name, kind=kind, device=device_for(cfg).type):
                return _train(cfg, kind, name, destination, resume, emit_predictions)
        except torch.cuda.OutOfMemoryError:
            if kind != "text" or cfg["text"]["batch_size"] <= 1:
                raise
            cfg["text"]["batch_size"] //= 2
            cfg["text"]["accumulation"] *= 2
            torch.cuda.empty_cache()
            # A failed first pass restarts from its seed; a durable checkpoint resumes.
            resume = (destination / "last.pt").exists()
            print(f"CUDA OOM: retry batch={cfg['text']['batch_size']} accumulation={cfg['text']['accumulation']}", flush=True)


def _train(cfg, kind, name, destination, resume, emit_predictions=True):
    seed_everything(cfg["seed"])
    device, settings = device_for(cfg), cfg[kind]
    latest = destination / "last.pt"
    resume_state = torch.load(latest, map_location="cpu", weights_only=False) if resume and latest.exists() else None
    if resume and resume_state is None:
        raise FileNotFoundError("No durable resume checkpoint")
    if resume_state:
        old = resume_state["cfg"]
        # Dataloader batching/accumulation must match for an exact mid-epoch resume.
        for key in ["batch_size", "workers", "learning_rate", "max_train_rows", "accumulation", "drop_title", "drop_user", "drop_parent", "ce_weight"]:
            if old[kind].get(key) != settings.get(key):
                raise ValueError(f"Resume configuration changed: {kind}.{key}; use a new model name")
        if old["seed"] != cfg["seed"]:
            raise ValueError("Cannot change seed when resuming")
    model, pad_id = build_model(cfg, kind, destination, initialize=resume_state is None)
    model.to(device)
    if kind == "text":
        groups = [{"params": model.encoder.parameters(), "lr": settings["learning_rate"]},
                  {"params": model.head.parameters(), "lr": settings["head_learning_rate"]}]
    else:
        groups = model.parameters()
    optimizer = torch.optim.AdamW(groups, lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    accumulation = settings.get("accumulation", 1)
    manifest = read_json(Path(cfg["run_dir"]) / "manifest.json")
    expected_rows = min(manifest["counts"]["train"], settings.get("max_train_rows") or manifest["counts"]["train"])
    expected_steps = max(1, math.ceil(expected_rows / settings["batch_size"] / accumulation) * settings["epochs"])
    warmup_steps = max(1, int(expected_steps * .06))
    def schedule(index):
        if kind != "text":
            return 1.
        if index < warmup_steps:
            return (index + 1) / warmup_steps
        return max(0., (expected_steps-index) / max(1, expected_steps-warmup_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    amp = kind == "text" and settings["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    epoch0, skip, step, best, stale, consumed = 0, 0, 0, math.inf, 0, 0.
    if resume_state:
        model.load_state_dict(resume_state["model"])
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
        scaler.load_state_dict(resume_state["scaler"])
        epoch0, skip, step = resume_state["epoch"], resume_state["batches_seen"], resume_state["step"]
        best, stale, consumed = resume_state["best"], resume_state["stale"], resume_state["consumed"]
        saved_rng = resume_state["rng"]
    else:
        saved_rng = None
    write_json(destination / "training_config.json", {"kind": kind, "name": name, "cfg": cfg})
    started = time.monotonic()
    limit = settings["max_hours"] * 3600
    last_calibration = None

    def elapsed():
        return consumed + time.monotonic() - started

    def save_last(epoch, batches_seen):
        _atomic_torch_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "scheduler": scheduler.state_dict(),
             "epoch": epoch, "batches_seen": batches_seen, "step": step, "best": best, "stale": stale,
             "consumed": elapsed(), "rng": _rng_state(), "cfg": cfg}, latest)

    def calibrate(epoch, batches_seen):
        nonlocal best, last_calibration
        with timed(cfg, "calibration", model=name, epoch=epoch):
            records, truth = predict_loader(model, kind, loader(cfg, kind, "calibration", pad_id=pad_id), device, amp)
            score = rmse(truth, [r["prediction"] for r in records])
        last_calibration = score
        if score < best:
            best = score
            _atomic_torch_save(model.state_dict(), destination / "best.pt")
            write_json(destination / "best.json", {"calibration_rmse": score, "epoch": epoch, "step": step})
        save_last(epoch, batches_seen)
        print(f"{name}: step={step} calibration_rmse={score:.6f} best={best:.6f}", flush=True)
        model.train()

    optimizer.zero_grad(set_to_none=True)
    stop = False
    for epoch in range(epoch0, settings["epochs"]):
        previous_epoch = destination / f"epoch_{epoch}.json"
        if previous_epoch.exists() and epoch > epoch0:
            duration_estimate = read_json(previous_epoch)["training_seconds"]
            if elapsed() + duration_estimate * 1.15 >= limit:
                print("Remaining budget is insufficient for another complete epoch; retaining best checkpoint", flush=True)
                break
        epoch_best_before = best
        rows, pending, batches_seen = 0, 0, 0
        synchronize()
        train_start = time.perf_counter()
        calibration_seconds = 0.
        resumed_partial = skip > 0
        model.train()
        for batch_index, batch in enumerate(loader(cfg, kind, "train", epoch=epoch, pad_id=pad_id)):
            if batch_index < skip:
                continue
            if saved_rng is not None:
                _restore_rng(saved_rng)
                saved_rng = None
            batches_seen = batch_index + 1
            rating = batch["rating"].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                prediction, logits = forward(model, kind, batch, device)
                loss = TextRatingModel.loss(prediction, logits, rating, settings["ce_weight"]) if kind == "text" else torch.nn.functional.mse_loss(prediction.float(), rating)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss; checkpoint retained")
            scaler.scale(loss / accumulation).backward()
            rows += len(batch["id"])
            pending += 1
            if pending < accumulation:
                continue
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), 1.)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            pending = 0
            step += 1
            interval = settings.get("evaluate_every_steps", 0)
            if interval and step % interval == 0:
                before = time.perf_counter()
                calibrate(epoch, batches_seen)
                calibration_seconds += time.perf_counter() - before
            if step % 100 == 0:
                print(f"{name}: epoch={epoch+1} step={step} rows={rows} loss={loss.item():.4f} hours={elapsed()/3600:.2f}", flush=True)
            if elapsed() >= limit:
                stop = True
                break
        skip = 0
        if pending:
            # Correct normalization for a final accumulation group shorter than K.
            scaler.unscale_(optimizer)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(accumulation / pending)
            clip_grad_norm_(model.parameters(), 1.)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
        synchronize()
        duration = time.perf_counter() - train_start - calibration_seconds
        if rows == 0 and not resumed_partial:
            raise ValueError("Training sample empty")
        write_json(destination / f"epoch_{epoch+1}.json", {"epoch": epoch+1, "training_seconds": duration,
                   "processed_rows": rows, "complete_epoch": not stop and not resumed_partial,
                   "resumed_segment": resumed_partial, "rows_per_second": rows/max(duration, 1e-9)})
        calibrate(epoch if stop else epoch + 1, batches_seen if stop else 0)
        stale = stale + 1 if best >= epoch_best_before else 0
        save_last(epoch if stop else epoch + 1, batches_seen if stop else 0)
        if stop or stale >= settings.get("patience", 2):
            break
    model.load_state_dict(torch.load(destination / "best.pt", map_location=device, weights_only=True))
    # Validation is generated only after calibration has selected the checkpoint.
    for split in (["calibration", "validation", "test"] if emit_predictions else ["calibration"]):
        with timed(cfg, "model_prediction", model=name, split=split):
            records, _ = predict_loader(model, kind, loader(cfg, kind, split, pad_id=pad_id), device, amp)
            write_predictions(Path(cfg["run_dir"]) / "predictions" / name / f"{split}.csv", records)
    return best


def pilot(cfg):
    """Two calibration-only loss comparisons, then a measured main-run budget."""
    import copy
    root = Path(cfg["run_dir"])
    results = []
    for value in [0., 0.2]:
        candidate = copy.deepcopy(cfg)
        candidate["text"].update(ce_weight=value, max_train_rows=cfg["pilot"]["rows"], epochs=1,
                                  max_hours=cfg["pilot"]["hours_per_candidate"])
        name = "pilot_ce_" + str(value).replace(".", "_")
        score = train(candidate, "text", name, emit_predictions=False)
        epoch = read_json(root / "models" / name / "epoch_1.json")
        results.append({"name": name, "ce_weight": value, "calibration_rmse": score, **epoch})
    chosen = min(results, key=lambda r: r["calibration_rmse"])
    cfg["text"]["ce_weight"] = chosen["ce_weight"]
    n = read_json(root / "manifest.json")["counts"]["train"]
    estimate = n / chosen["rows_per_second"]
    # Leave time for evaluation/checkpoint IO; a hash sample preserves class mix.
    feasible = int(chosen["rows_per_second"] * cfg["text"]["max_hours"] * 3600 * 0.75)
    if feasible < n:
        cfg["text"]["max_train_rows"] = max(1, feasible)
    write_json(root / "pilot_selection.json", {"candidates": results, "selected_ce_weight": chosen["ce_weight"],
               "estimated_full_epoch_training_seconds": estimate, "main_max_train_rows": cfg["text"]["max_train_rows"]})
    write_json(root / "selected_config.json", cfg)
    return cfg


def predict(cfg, name, splits=("test",)):
    destination = Path(cfg["run_dir"]) / "models" / name
    spec = read_json(destination / "training_config.json")
    model_cfg, kind = spec["cfg"], spec["kind"]
    # Current root/device can differ when moving a completed run to another machine.
    model_cfg["run_dir"], model_cfg["device"] = cfg["run_dir"], cfg["device"]
    if kind == "text":
        cache_tokens(model_cfg, splits, force="test" in splits)
    device = device_for(model_cfg)
    model, pad = build_model(model_cfg, kind, destination, initialize=False)
    model.load_state_dict(torch.load(destination / "best.pt", map_location="cpu", weights_only=True))
    model.to(device)
    for split in splits:
        records, _ = predict_loader(model, kind, loader(model_cfg, kind, split, pad_id=pad), device,
                                    kind == "text" and model_cfg["text"]["amp"])
        write_predictions(Path(cfg["run_dir"]) / "predictions" / name / f"{split}.csv", records)
