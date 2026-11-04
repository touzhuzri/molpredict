"""Train fresh models on the merged fungicide dataset.

A: SMILES baseline
B: SMILES + 24 PaDEL
C: B + SMILES -> 24 PaDEL auxiliary learning on all parseable structures
B_OLD_ONLY: source ablation (new-only positives removed from training)

The test split is touched once, after early stopping and threshold selection
on validation data.  No old classifier checkpoint is loaded.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    CHECKPOINTS, DATA, FEATURES, LOGS, METRICS, SPLITS,
    best_threshold, binary_metrics, dump_json,
)
from modeling import (  # noqa: E402
    AuxDataset, MergedDataset, build_model, collate_aux, collate_merged,
    count_parameters, masked_mse, move_aux, move_batch,
)
from molpredict.v3_data import NumericStandardizer  # noqa: E402
from molpredict.v3_model import V3Config  # noqa: E402

MODEL_CONFIGS = {
    "A": dict(modalities=("smiles",), aux=False),
    "B": dict(modalities=("smiles", "descriptors"), aux=False),
    "C": dict(modalities=("smiles", "descriptors"), aux=True),
    "B_OLD_ONLY": dict(modalities=("smiles", "descriptors"), aux=False),
}


def resolve_device(device_name: str):
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def gpu_info(device: torch.device) -> dict:
    if device.type != "cuda":
        return {"available": False, "torch": torch.__version__}
    prop = torch.cuda.get_device_properties(device.index or 0)
    return {
        "available": True,
        "name": prop.name,
        "total_memory_gb": round(prop.total_memory / 1024 ** 3, 2),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def warmup_cosine(optimizer, total_steps, warmup_ratio, min_lr_ratio):
    warmup = max(1, int(total_steps * warmup_ratio))

    def fn(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(total_steps - warmup, 1)
        progress = min(max(progress, 0.0), 1.0)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (
            1 + math.cos(math.pi * progress)
        )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


def make_loader(ds, batch_size, shuffle, seed):
    gen = torch.Generator()
    gen.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate_merged, generator=gen,
                      pin_memory=torch.cuda.is_available())


def make_aux_loader(ds, batch_size, seed):
    gen = torch.Generator()
    gen.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      collate_fn=collate_aux, generator=gen,
                      pin_memory=torch.cuda.is_available())


def evaluate(model, loader, criterion, config, device):
    model.eval()
    probs, labels, losses = [], [], []
    with torch.no_grad():
        for raw in loader:
            batch = move_batch(raw, config, device)
            with torch.autocast(device_type=device.type,
                                dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                logits = model(batch)
            loss = criterion(logits.float(), batch["label"])
            probs.extend(torch.sigmoid(logits.float()).cpu().numpy().tolist())
            labels.extend(batch["label"].cpu().numpy().tolist())
            losses.append(float(loss.item()) * batch["label"].size(0))
    y = np.asarray(labels).astype(int)
    p = np.asarray(probs, dtype=float)
    out = {
        "labels": y, "probs": p,
        "loss": float(sum(losses) / max(len(y), 1)),
    }
    if len(np.unique(y)) == 2:
        out["roc_auc"] = float(roc_auc_score(y, p))
        out["pr_auc"] = float(average_precision_score(y, p))
    else:
        out["roc_auc"] = None
        out["pr_auc"] = None
    return out


def _row_metrics(df, probs, mask, threshold):
    y = df.loc[mask, "label"].to_numpy(dtype=int)
    p = probs[mask]
    if len(y) == 0:
        return {"n": 0, "positives": 0, "note": "empty subset"}
    return binary_metrics(y, p, threshold)


def run_one(name: str, split_seed: int, seed: int, args) -> dict:
    seed_everything(seed)
    device = resolve_device(args.device)
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("formal experiment requires CUDA")
    cfg = MODEL_CONFIGS[name]
    model_cfg = V3Config(
        modalities=cfg["modalities"], d_model=128, n_layers=4, n_heads=4,
        ff_mult=2, dropout=args.dropout, max_smiles_len=512,
        n_descriptors=24, pooling="cls",
    )
    aux = bool(cfg["aux"])

    table = __import__("pandas").read_csv(
        DATA / f"modeling_structures_seed{split_seed}.csv", low_memory=False
    ).sort_values("model_index").reset_index(drop=True)
    split = json.loads(
        (SPLITS / f"split_seed{split_seed}.json").read_text(encoding="utf-8")
    )
    train_idx = np.asarray(split["train"], dtype=int)
    val_idx = np.asarray(split["val"], dtype=int)
    test_idx = np.asarray(split["test"], dtype=int)
    if name == "B_OLD_ONLY":
        # Keep only structures that have an old PPDB positive/negative label.
        keep = (table["old_positive"] == 1) | (table["old_negative"] == 1)
        train_idx = train_idx[keep.to_numpy()[train_idx]]

    struct = __import__("pandas").read_csv(
        DATA / "structures.csv", low_memory=False
    ).sort_values("structure_index").reset_index(drop=True)
    if "descriptors" in model_cfg.modalities:
        raw_desc = np.load(FEATURES / "padel24.npy")
    else:
        # A does not consume PaDEL; keep the array shape valid.
        raw_desc = np.zeros((len(struct), 24), dtype=np.float32)
    states = torch.load(FEATURES / "smiles_states.pt",
                        weights_only=False)
    tokens = torch.load(FEATURES / "smiles_tokens.pt",
                        weights_only=False)
    if states["structure_ids"] != struct["structure_id"].tolist():
        raise ValueError("SMILES cache order does not match structures.csv")
    if tokens["structure_ids"] != struct["structure_id"].tolist():
        raise ValueError("SMILES token order does not match structures.csv")
    if raw_desc.shape != (len(struct), 24):
        raise ValueError("PaDEL matrix shape mismatch")

    standardizer = NumericStandardizer.fit(raw_desc, train_idx_feature(table, train_idx))
    descriptors = standardizer.transform(raw_desc)
    arrays = {
        "smiles_states": states["states"],
        "smiles_ids": tokens["input_ids"],
        "descriptors": descriptors,
        "raw_descriptors": raw_desc,
    }
    labels = table["label"].to_numpy(dtype=np.float32)
    feature_indices = table["structure_index"].to_numpy(dtype=int)

    ds = {
        "train": MergedDataset(train_idx, labels, feature_indices, arrays),
        "val": MergedDataset(val_idx, labels, feature_indices, arrays),
        "test": MergedDataset(test_idx, labels, feature_indices, arrays),
    }
    loaders = {
        k: make_loader(v, args.batch_size, k == "train", seed)
        for k, v in ds.items()
    }
    aux_loader = None
    n_aux = 0
    if aux:
        # Unlabeled/conflict structures can help the auxiliary task, but
        # neither validation nor test scaffolds may enter any training
        # signal.  This set contains train structures plus unlabeled /
        # conflict structures whose scaffold is disjoint from val/test.
        valtest_scaffolds = set(table.iloc[val_idx]["scaffold"]) | set(
            table.iloc[test_idx]["scaffold"]
        )
        aux_indices = struct.index[
            ~struct["scaffold"].isin(valtest_scaffolds)
        ].to_numpy(dtype=int)
        aux_ds = AuxDataset(aux_indices, arrays)
        aux_loader = make_aux_loader(aux_ds, args.batch_size, seed)
        n_aux = len(aux_ds)

    model = build_model(model_cfg, aux=aux).to(device)
    total_params, trainable_params = count_parameters(model)
    pos = float(labels[train_idx].sum())
    neg = float(len(train_idx) - pos)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([neg / max(pos, 1.0)], device=device)
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    steps_per_epoch = math.ceil(len(loaders["train"]) / args.grad_accum)
    scheduler = warmup_cosine(
        optimizer, max(steps_per_epoch * args.epochs, 1),
        args.warmup_ratio, args.min_lr_ratio,
    )

    run_dir = CHECKPOINTS / name / f"split{split_seed}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "experiment": name,
        "model": model_cfg.to_dict(),
        "aux": aux,
        "aux_weight": args.aux_weight if aux else 0.0,
        "aux_data": ("train structures + unlabeled/conflict structures "
                     "whose scaffold is disjoint from val/test") if aux else None,
        "n_aux_structures": n_aux,
        "split_seed": split_seed,
        "seed": seed,
        "training": {
            "train_subset": "old_labeled_only" if name == "B_OLD_ONLY"
            else "all_labeled_merged",
            "epochs": args.epochs, "patience": args.patience,
            "batch_size": args.batch_size, "grad_accum": args.grad_accum,
            "lr": args.lr, "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "min_lr_ratio": args.min_lr_ratio, "amp": "bf16",
        },
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "n_train_positive": int(labels[train_idx].sum()),
        "n_train_negative": int(len(train_idx) - labels[train_idx].sum()),
    }
    dump_json(run_dir / "config.json", run_config)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(run_config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    dump_json(run_dir / "standardization.json", standardizer.to_dict())
    dump_json(run_dir / "split.json", {
        "train": train_idx.tolist(), "val": val_idx.tolist(),
        "test": test_idx.tolist(), "split_seed": split_seed,
    })

    history = []
    best_state, best_score, best_epoch = None, -1.0, -1
    bad = 0
    aux_iter = None
    if aux_loader is not None:
        def cycle():
            while True:
                for b in aux_loader:
                    yield b
        aux_iter = cycle()
    started = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        total_loss, seen = 0.0, 0
        aux_total, aux_seen = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        n_batches = len(loaders["train"])
        for step, raw in enumerate(loaders["train"], start=1):
            batch = move_batch(raw, model_cfg, device)
            with torch.autocast(device_type=device.type,
                                dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                logits = model(batch)
            raw_loss = criterion(logits.float(), batch["label"])
            if aux_iter is not None:
                aux_batch = move_aux(next(aux_iter), device)
                with torch.autocast(device_type=device.type,
                                    dtype=torch.bfloat16,
                                    enabled=device.type == "cuda"):
                    aux_pred = model.forward_aux(aux_batch)
                aux_loss = masked_mse(
                    aux_pred, aux_batch["aux_target"], aux_batch["aux_mask"]
                )
                loss = (raw_loss + args.aux_weight * aux_loss) / args.grad_accum
                aux_total += float(aux_loss.item()) * batch["label"].size(0)
                aux_seen += batch["label"].size(0)
            else:
                aux_loss = None
                loss = raw_loss / args.grad_accum
            loss.backward()
            total_loss += float(raw_loss.item()) * batch["label"].size(0)
            seen += batch["label"].size(0)
            if step % args.grad_accum == 0 or step == n_batches:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    args.grad_clip,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        train_eval = evaluate(model, loaders["train"], criterion, model_cfg,
                              device)
        val_eval = evaluate(model, loaders["val"], criterion, model_cfg,
                            device)
        epoch_sec = time.time() - epoch_start
        peak_mb = (torch.cuda.max_memory_allocated(device) / 1024 ** 2
                   if device.type == "cuda" else 0.0)
        entry = {
            "epoch": epoch,
            "train_loss": total_loss / max(seen, 1),
            "train_roc_auc": train_eval["roc_auc"],
            "train_pr_auc": train_eval["pr_auc"],
            "val_loss": val_eval["loss"],
            "val_roc_auc": val_eval["roc_auc"],
            "val_pr_auc": val_eval["pr_auc"],
            "aux_loss": (aux_total / max(aux_seen, 1)
                         if aux_iter is not None else None),
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_sec": round(epoch_sec, 2),
            "peak_allocated_mb": round(peak_mb, 1),
            "elapsed_sec": round(time.time() - started, 1),
        }
        history.append(entry)
        dump_json(run_dir / "history.json", history)
        print(f"[{name}/split{split_seed}/seed{seed}] epoch {epoch} "
              f"train_auc={train_eval['roc_auc']:.4f} "
              f"val_auc={val_eval['roc_auc']:.4f} "
              f"val_pr={val_eval['pr_auc']:.4f} "
              + (f"aux={entry['aux_loss']:.4f} " if aux_iter else "")
              + f"{epoch_sec:.1f}s", flush=True)

        score = val_eval["roc_auc"]
        if score > best_score + 1e-6:
            best_score, best_epoch, bad = score, epoch, 0
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[{name}/seed{seed}] early stop at {epoch}, "
                      f"best {best_epoch}", flush=True)
                break

    if best_state is None:
        raise RuntimeError("no checkpoint was produced")
    model.load_state_dict(best_state)
    model.to(device)
    val_eval = evaluate(model, loaders["val"], criterion, model_cfg, device)
    threshold = best_threshold(val_eval["labels"], val_eval["probs"])
    val_metrics = binary_metrics(val_eval["labels"], val_eval["probs"],
                                 threshold)
    train_eval = evaluate(model, loaders["train"], criterion, model_cfg, device)
    test_eval = evaluate(model, loaders["test"], criterion, model_cfg, device)
    test_metrics = binary_metrics(test_eval["labels"], test_eval["probs"],
                                  threshold)
    test_metrics["loss"] = test_eval["loss"]

    test_df = table.iloc[test_idx].reset_index(drop=True)
    probs = test_eval["probs"]
    y = test_df["label"].to_numpy(dtype=int)
    negatives = y == 0
    old_pos = (y == 1) & (test_df["old_positive"].to_numpy() == 1)
    new_pos = (y == 1) & (test_df["old_positive"].to_numpy() == 0)
    multi_pos = (y == 1) & (test_df["is_multi_component"].to_numpy() == 1)
    single_pos = (y == 1) & (test_df["is_multi_component"].to_numpy() == 0)
    old_structures = test_df["n_old_records"].to_numpy() > 0

    subset = {
        "all_test": test_metrics,
        "old_positive_vs_negative": _row_metrics(
            test_df, probs, negatives | old_pos, threshold
        ),
        "new_only_positive_vs_negative": _row_metrics(
            test_df, probs, negatives | new_pos, threshold
        ),
        "multi_component_positive_vs_negative": _row_metrics(
            test_df, probs, negatives | multi_pos, threshold
        ),
        "single_component_positive_vs_negative": _row_metrics(
            test_df, probs, negatives | single_pos, threshold
        ),
        "old_structures_only": _row_metrics(
            test_df, probs, old_structures, threshold
        ),
    }
    predictions = __import__("pandas").DataFrame({
        "structure_id": test_df["structure_id"],
        "label": y,
        "prob": probs,
        "old_positive": test_df["old_positive"].to_numpy(dtype=int),
        "new_positive": test_df["new_positive"].to_numpy(dtype=int),
        "new_only_positive": new_pos.astype(int),
        "is_multi_component": test_df["is_multi_component"].to_numpy(dtype=int),
        "n_old_records": test_df["n_old_records"].to_numpy(dtype=int),
        "n_new_records": test_df["n_new_records"].to_numpy(dtype=int),
        "scaffold": test_df["scaffold"],
    })
    predictions.to_csv(run_dir / "predictions_test.csv", index=False,
                       encoding="utf-8-sig")
    torch.save(best_state, run_dir / "model.pt")
    peak_alloc = max(float(h["peak_allocated_mb"]) for h in history)
    metrics = {
        "experiment": name,
        "split_seed": split_seed,
        "seed": seed,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "n_train_positive": int(labels[train_idx].sum()),
        "n_train_negative": int(len(train_idx) - labels[train_idx].sum()),
        "n_params_total": total_params,
        "n_params_trainable": trainable_params,
        "best_epoch": int(best_epoch),
        "epochs_run": len(history),
        "threshold": float(threshold),
        "val": val_metrics,
        "train": binary_metrics(train_eval["labels"], train_eval["probs"],
                                threshold),
        "test": test_metrics,
        "test_subsets": subset,
        "aux": {
            "enabled": aux,
            "weight": args.aux_weight if aux else 0.0,
            "n_aux_structures": n_aux,
            "final_loss": history[-1]["aux_loss"] if aux else None,
        },
        "train_test_auc_gap": (
            binary_metrics(train_eval["labels"], train_eval["probs"],
                           threshold)["roc_auc"] - test_metrics["roc_auc"]
            if test_metrics["roc_auc"] is not None else None
        ),
        "elapsed_sec": round(time.time() - started, 1),
        "mean_epoch_sec": round(
            float(np.mean([h["epoch_sec"] for h in history])), 2
        ),
        "peak_allocated_mb": peak_alloc,
        "gpu": gpu_info(device),
    }
    dump_json(run_dir / "metrics.json", metrics)
    return metrics


def train_idx_feature(table, train_idx):
    return table["structure_index"].to_numpy(dtype=int)[train_idx]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", default="A,B,C")
    parser.add_argument("--split-seeds", default="42")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--aux-weight", type=float, default=0.2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--require-cuda", action="store_true", default=True)
    args = parser.parse_args()

    names = [x.strip() for x in args.experiments.split(",") if x.strip()]
    split_seeds = [int(x) for x in args.split_seeds.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    for name in names:
        if name not in MODEL_CONFIGS:
            raise SystemExit(f"unknown experiment {name}")
    all_metrics = []
    for name in names:
        for split_seed in split_seeds:
            for seed in seeds:
                metrics = run_one(name, split_seed, seed, args)
                all_metrics.append(metrics)
                dump_json(METRICS / f"{name}_split{split_seed}_seed{seed}.json",
                          metrics)
    # Update a compact summary after every invocation.
    rows = []
    for m in all_metrics:
        rows.append({
            "experiment": m["experiment"], "split_seed": m["split_seed"],
            "seed": m["seed"], "test_roc_auc": m["test"]["roc_auc"],
            "test_pr_auc": m["test"]["pr_auc"], "test_f1": m["test"]["f1"],
            "test_mcc": m["test"]["mcc"],
            "val_roc_auc": m["val"]["roc_auc"], "best_epoch": m["best_epoch"],
            "epochs_run": m["epochs_run"],
            "n_params_total": m["n_params_total"],
            "elapsed_sec": m["elapsed_sec"],
            "peak_allocated_mb": m["peak_allocated_mb"],
        })
    if rows:
        with (METRICS / "runs.csv").open("w", encoding="utf-8-sig",
                                         newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
