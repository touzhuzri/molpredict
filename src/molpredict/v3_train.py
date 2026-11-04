"""V3 训练、评估与 checkpoint 保存。"""

from __future__ import annotations

import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from molpredict.padel import PADEL_NAMES
from molpredict.v2 import best_threshold, binary_metrics
from molpredict.v3_data import (
    NumericStandardizer,
    V3Dataset,
    collate_v3,
    load_cache_arrays,
    load_rows,
    load_split,
)
from molpredict.v3_encoders import (
    load_confseq_parts,
    load_smiles_encoder,
    pretrained_manifest,
    save_confseq_vocab,
    save_tokenizer,
    sha256,
)
from molpredict.v3_model import V3Config, V3Net, count_trainable

ROOT = Path(__file__).resolve().parents[2]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2),
                    encoding="utf-8")


def _loaders(split, labels, arrays, standardizer, batch_size):
    descriptors = standardizer.transform(arrays["descriptors"])
    datasets = {
        k: V3Dataset(
            split[k], labels,
            smiles_states=arrays["smiles_states"],
            smiles_ids=arrays["smiles_ids"],
            confseq_states=arrays["confseq_states"],
            confseq_ids=arrays["confseq_ids"],
            descriptors=descriptors,
        ) for k in ("train", "val", "test")
    }
    return {
        "train": DataLoader(datasets["train"], batch_size=batch_size,
                            shuffle=True, collate_fn=collate_v3),
        "val": DataLoader(datasets["val"], batch_size=batch_size,
                          shuffle=False, collate_fn=collate_v3),
        "test": DataLoader(datasets["test"], batch_size=batch_size,
                           shuffle=False, collate_fn=collate_v3),
    }


def _forward_probs(model, loader):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch)
            probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            labels.extend(batch["label"].cpu().numpy().tolist())
    return np.asarray(probs), np.asarray(labels).astype(int)


def build_optimizer(model: V3Net, lr: float, encoder_lr: float,
                    weight_decay: float):
    encoder_params = []
    for name in ("smiles_encoder", "confseq_encoder"):
        module = getattr(model, name, None)
        if module is None:
            continue
        encoder_params.extend(
            [p for p in module.parameters() if p.requires_grad]
        )
    encoder_ids = {id(p) for p in encoder_params}
    head_params = [
        p for p in model.parameters()
        if p.requires_grad and id(p) not in encoder_ids
    ]
    groups = [{"params": head_params, "lr": lr}]
    if encoder_params:
        groups.append({"params": encoder_params, "lr": encoder_lr})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def train_v3(config: V3Config, split: dict, cache_dir: Path, out_dir: Path,
             seed: int, *, epochs: int = 40, patience: int = 8,
             batch_size: int = 16, lr: float = 1e-3,
             encoder_lr: float = 2e-5, weight_decay: float = 1e-4,
             quick_limit: int | None = None) -> dict:
    """训练一个 V3 配置并保存完整产物。

    训练、验证、测试分别来自固定 scaffold 划分；测试集只在此函数末尾评估一次。
    """
    seed_everything(seed)
    n_threads = int(os.environ.get("MOLPREDICT_THREADS", "0"))
    if n_threads > 0:
        torch.set_num_threads(n_threads)
    out_dir.mkdir(parents=True, exist_ok=True)
    mols, labels, rows = load_rows()
    expected = [r["inchikey"] for r in rows]
    arrays = load_cache_arrays(
        ROOT / "experiments" / "v2" / "cache", cache_dir, expected
    )
    split = {k: np.asarray(v, dtype=int) for k, v in split.items()}
    if quick_limit:
        # 仅用于冒烟测试：截取训练/验证/测试的前若干条，不用于正式实验
        split = {k: v[:quick_limit] for k, v in split.items()}

    standardizer = NumericStandardizer.fit(
        arrays["descriptors"], split["train"]
    )
    loaders = _loaders(split, labels, arrays, standardizer, batch_size)

    model = V3Net(config)
    total_params, trainable_params = count_trainable(model)
    pos = float(labels[split["train"]].sum())
    neg = float(len(split["train"]) - pos)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)
    )
    optimizer = build_optimizer(model, lr, encoder_lr, weight_decay)

    best_state, best_score, best_epoch, bad = None, -1.0, -1, 0
    history = []
    started = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, seen = 0.0, 0
        for batch in loaders["train"]:
            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits, batch["label"])
            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 5.0
            )
            optimizer.step()
            total_loss += float(loss.item()) * len(batch["label"])
            seen += len(batch["label"])
        train_loss = total_loss / max(seen, 1)

        val_probs, val_true = _forward_probs(model, loaders["val"])
        val_auc = (roc_auc_score(val_true, val_probs)
                   if len(np.unique(val_true)) == 2 else 0.0)
        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_roc_auc": float(val_auc),
        })
        print(f"[{config.modalities}/seed{seed}] epoch {epoch} "
              f"loss={train_loss:.4f} val_auc={val_auc:.4f}", flush=True)
        if val_auc > best_score:
            best_score, best_epoch, bad = val_auc, epoch, 0
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is None:
        raise RuntimeError("训练没有产生可用的 checkpoint")
    model.load_state_dict(best_state)
    val_probs, val_true = _forward_probs(model, loaders["val"])
    threshold = best_threshold(val_true, val_probs)
    val_metrics = binary_metrics(val_true, val_probs, threshold)
    test_probs, test_true = _forward_probs(model, loaders["test"])
    test_metrics = binary_metrics(test_true, test_probs, threshold)

    # 保存权重、结构、tokenizer、标准化参数、特征顺序、来源和拆分
    torch.save(best_state, out_dir / "model.pt")
    _save_json(out_dir / "standardization.json", standardizer.to_dict())
    _save_json(out_dir / "feature_order.json", {
        "padel24": PADEL_NAMES,
        "n": len(PADEL_NAMES),
    })
    if "smiles" in config.modalities:
        save_tokenizer(out_dir / "tokenizer")
    if "confseq" in config.modalities:
        save_confseq_vocab(out_dir)
    _save_json(out_dir / "split.json",
               {k: np.asarray(v).tolist() for k, v in split.items()})
    _save_json(out_dir / "pretrained.json", pretrained_manifest())

    cache_manifest_path = cache_dir / "manifest.json"
    cache_manifest = (
        json.loads(cache_manifest_path.read_text(encoding="utf-8"))
        if cache_manifest_path.exists() else {}
    )
    config_payload = config.to_dict()
    config_payload.update({
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_max": epochs,
        "patience": patience,
        "batch_size": batch_size,
        "lr": lr,
        "encoder_lr": encoder_lr,
        "weight_decay": weight_decay,
        "n_train": int(len(split["train"])),
        "n_val": int(len(split["val"])),
        "n_test": int(len(split["test"])),
        "n_train_pos": int(labels[split["train"]].sum()),
        "n_val_pos": int(labels[split["val"]].sum()),
        "n_test_pos": int(labels[split["test"]].sum()),
        "threshold": float(threshold),
        "threshold_source": "validation set, max F1",
        "training_csv": str(ROOT / "data" / "training" / "confseq.csv"),
        "training_csv_sha256": sha256(
            ROOT / "data" / "training" / "confseq.csv"
        ),
        "cache_manifest": cache_manifest,
        "pretrained": pretrained_manifest(),
        "n_params_total": int(total_params),
        "n_params_trainable": int(trainable_params),
        "elapsed_sec": round(time.time() - started, 1),
        "label_definition": (
            "PPDB fungicide classification (1=fungicide list, "
            "0=herbicide/insecticide list); not experimental activity"
        ),
    })
    _save_json(out_dir / "config.json", config_payload)
    _save_json(out_dir / "metrics.json", {
        "val": val_metrics, "test": test_metrics, "history": history,
    })
    with (out_dir / "predictions_test.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["name", "smiles", "label", "probability",
                         "threshold", "predicted"])
        for idx, p in zip(split["test"], test_probs):
            writer.writerow([
                rows[idx].get("name", ""), rows[idx]["smiles"],
                int(labels[idx]), f"{p:.6f}", f"{threshold:.6f}",
                int(p >= threshold),
            ])
    print(f"[V3/{','.join(config.modalities)}/seed{seed}] "
          f"test AUC={test_metrics['roc_auc']:.4f} "
          f"PR-AUC={test_metrics['pr_auc']:.4f} "
          f"F1={test_metrics['f1']:.4f} MCC={test_metrics['mcc']:.4f} "
          f"(best epoch {best_epoch}, {time.time()-started:.0f}s)", flush=True)
    return {"val": val_metrics, "test": test_metrics, "history": history,
            "best_epoch": best_epoch, "threshold": float(threshold),
            "n_params_total": total_params,
            "n_params_trainable": trainable_params,
            "elapsed_sec": round(time.time() - started, 1)}


def load_model_for_inference(model_dir: Path, device: str = "cpu"):
    """从 checkpoint 重建融合模型；调用方负责外部冻结编码器/预测。"""
    config = V3Config.from_dict(
        json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    )
    model = V3Net(config)
    state = torch.load(model_dir / "model.pt", map_location=device,
                       weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model, config


__all__ = [
    "train_v3", "load_model_for_inference", "build_optimizer",
    "save_tokenizer", "save_confseq_vocab", "load_smiles_encoder",
    "load_confseq_parts",
]
