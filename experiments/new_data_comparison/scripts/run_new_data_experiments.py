"""五组模型 A–E 的新数据对照实验入口。

A: SMILES
B: 24 PaDEL
C: SMILES + 24 PaDEL
D: C + structure→PaDEL 辅助任务（OLD_ONLY / NEW_AUX_MAIN / NEW_AUX_FULL）
E: C + 8 层 Transformer

A/B/C/E 的 NEW_AUX 条件记录为 N/A：它们没有辅助任务，无法合理使用新数据。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import ConcatDataset, DataLoader

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from molpredict.new_data_model import AuxDataset, AuxV3Net, collate_aux  # noqa: E402
from molpredict.v3_1_train import train_v3_1  # noqa: E402
from molpredict.v3_data import (  # noqa: E402
    NumericStandardizer,
    load_cache_arrays,
    load_rows,
    load_split,
)
from molpredict.v3_model import V3Config  # noqa: E402

OUT = ROOT / "experiments" / "new_data_comparison"
AUX_PT = OUT / "data_aux" / "new_aux.pt"
V2_CACHE = ROOT / "experiments" / "v2" / "cache"
V3_CACHE = ROOT / "experiments" / "v3" / "cache"

BASE_KW = dict(d_model=128, n_heads=4, ff_mult=2, dropout=0.3,
               pooling="cls")
CONFIGS = {
    "A_OLD_ONLY": ("A", V3Config(modalities=("smiles",), **BASE_KW), None),
    "B_OLD_ONLY": ("B", V3Config(modalities=("descriptors",), **BASE_KW), None),
    "C_OLD_ONLY": ("C", V3Config(modalities=("smiles", "descriptors"),
                                 **BASE_KW), None),
    "D_OLD_ONLY": ("D", V3Config(modalities=("smiles", "descriptors"),
                                 **BASE_KW), "old"),
    "D_NEW_AUX_MAIN": ("D", V3Config(modalities=("smiles", "descriptors"),
                                     **BASE_KW), "main"),
    "D_NEW_AUX_FULL": ("D", V3Config(modalities=("smiles", "descriptors"),
                                     **BASE_KW), "full"),
    "E_OLD_ONLY": ("E", V3Config(modalities=("smiles", "descriptors"),
                                 n_layers=8, **BASE_KW), None),
}
FIELDS = ["experiment", "seed", "arch", "modalities", "aux_source", "n_aux",
          "n_params_total", "n_params_trainable", "best_epoch", "epochs_run",
          "val_roc_auc", "val_pr_auc", "test_roc_auc", "test_pr_auc",
          "test_f1", "test_precision", "test_recall", "test_mcc",
          "train_roc_auc", "train_pr_auc", "train_test_auc_gap",
          "aux_loss_final", "threshold", "test_tn", "test_fp", "test_fn",
          "test_tp", "mean_epoch_sec", "peak_allocated_mb", "elapsed_sec"]


def _aux_arrays(arrays, labels, standardizer, split):
    raw = arrays["descriptors"]
    transformed = standardizer.transform(raw)
    mask = np.isfinite(raw).astype(np.float32)
    idx = np.asarray(split["train"], dtype=int)
    states = [arrays["smiles_states"][i].float() for i in idx]
    return AuxDataset(states, transformed[idx], mask[idx])


def _new_aux_arrays(standardizer, source: str):
    payload = torch.load(AUX_PT, weights_only=False)
    raw = np.asarray(payload["descriptors"], dtype=np.float32)
    transformed = standardizer.transform(raw)
    mask = np.isfinite(raw).astype(np.float32)
    strict = np.asarray(payload["strict"], dtype=bool)
    tier = np.asarray(payload["tier"])
    if source == "main":
        select = strict & (tier == "MAIN")
    elif source == "full":
        select = strict
    else:
        raise ValueError(source)
    idx = np.nonzero(select)[0]
    states = [payload["states"][i].float() for i in idx]
    return AuxDataset(states, transformed[idx], mask[idx]), len(idx)


def _build_aux_loader(arrays, labels, standardizer, split, source, seed,
                      batch_size=16):
    old = _aux_arrays(arrays, labels, standardizer, split)
    datasets = [old]
    n_new = 0
    if source in ("main", "full"):
        new, n_new = _new_aux_arrays(standardizer, source)
        datasets.append(new)
    ds = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    gen = torch.Generator()
    gen.manual_seed(seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        collate_fn=collate_aux, generator=gen)
    return loader, len(ds), n_new


def _row(name, arch, config, aux_source, n_aux, seed, metrics, patience):
    t, v, tr = metrics["test"], metrics["val"], metrics["train"]
    hist = metrics["history"]
    peak = max((float(h["gpu"].get("peak_allocated_mb", 0) or 0)
                for h in hist), default=0.0)
    mean_epoch = sum(float(h["epoch_sec"]) for h in hist) / max(len(hist), 1)
    aux_last = hist[-1].get("aux_loss") if hist else None
    return {
        "experiment": name, "seed": seed, "arch": arch,
        "modalities": "+".join(config.modalities),
        "aux_source": aux_source or "", "n_aux": n_aux,
        "n_params_total": metrics["n_params_total"],
        "n_params_trainable": metrics["n_params_trainable"],
        "best_epoch": metrics["best_epoch"],
        "epochs_run": len(hist),
        "val_roc_auc": v["roc_auc"], "val_pr_auc": v["pr_auc"],
        "test_roc_auc": t["roc_auc"], "test_pr_auc": t["pr_auc"],
        "test_f1": t["f1"], "test_precision": t["precision"],
        "test_recall": t["recall"], "test_mcc": t["mcc"],
        "train_roc_auc": tr["roc_auc"], "train_pr_auc": tr["pr_auc"],
        "train_test_auc_gap": tr["roc_auc"] - t["roc_auc"],
        "aux_loss_final": aux_last,
        "threshold": t["threshold"],
        "test_tn": t["confusion_matrix"]["tn"],
        "test_fp": t["confusion_matrix"]["fp"],
        "test_fn": t["confusion_matrix"]["fn"],
        "test_tp": t["confusion_matrix"]["tp"],
        "mean_epoch_sec": round(mean_epoch, 2),
        "peak_allocated_mb": round(peak, 1),
        "elapsed_sec": metrics["elapsed_sec"],
    }


def _write_summary(out_dir: Path, rows):
    with (out_dir / "summary.csv").open("w", encoding="utf-8-sig",
                                        newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    groups = defaultdict(list)
    for row in rows:
        groups[row["experiment"]].append(row)
    stability = {}
    for name, items in groups.items():
        block = {"n_runs": len(items), "aux_source": items[0]["aux_source"],
                 "n_aux": items[0]["n_aux"]}
        for metric in ("test_roc_auc", "test_pr_auc", "test_f1",
                       "test_mcc", "val_roc_auc", "train_roc_auc",
                       "train_test_auc_gap"):
            vals = [float(x[metric]) for x in items]
            mean = sum(vals) / len(vals)
            std = (sum((v - mean) ** 2 for v in vals)
                   / max(len(vals) - 1, 1)) ** 0.5
            block[metric] = {"mean": mean, "std": std, "values": vals}
        block["best_epochs"] = [int(x["best_epoch"]) for x in items]
        stability[name] = block
    (out_dir / "stability.json").write_text(
        json.dumps(stability, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return stability


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", default=",".join(CONFIGS))
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--aux-weight", type=float, default=0.2)
    parser.add_argument("--amp", default="bf16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    out_dir = Path(args.out)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (out_dir / "configs").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    names = [s.strip() for s in args.experiments.split(",") if s.strip()]
    unknown = [n for n in names if n not in CONFIGS]
    if unknown:
        raise SystemExit(f"unknown experiments: {unknown}")

    mols, labels, rows = load_rows()
    expected = [r["inchikey"] for r in rows]
    arrays = load_cache_arrays(V2_CACHE, V3_CACHE, expected)

    records = []
    for name in names:
        arch, config, aux_source = CONFIGS[name]
        if args.dropout != config.dropout:
            payload = config.to_dict()
            payload["dropout"] = args.dropout
            config = V3Config.from_dict(payload)
        (out_dir / "configs" / f"{name}.yaml").write_text(
            yaml.safe_dump({
                "experiment": name, "arch": arch,
                "model": config.to_dict(), "aux_source": aux_source,
                "aux_weight": args.aux_weight,
                "training": {
                    "epochs": args.epochs, "patience": args.patience,
                    "batch_size": args.batch_size,
                    "grad_accum": args.grad_accum, "lr": args.lr,
                    "amp": args.amp, "device": args.device,
                    "label": "old PPDB classification only",
                },
                "note": ("A/B/C/E NEW_AUX=N/A: no auxiliary objective; "
                         "new data not usable without changing the model."),
            }, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        for seed in seeds:
            split = load_split(seed)
            standardizer = NumericStandardizer.fit(
                arrays["descriptors"], split["train"]
            )
            run_dir = out_dir / "checkpoints" / f"{name}_seed{seed}"
            aux_loader = None
            n_aux = 0
            n_new = 0
            model = None
            if aux_source is not None:
                aux_loader, n_aux, n_new = _build_aux_loader(
                    arrays, labels, standardizer, split, aux_source, seed,
                    batch_size=args.batch_size,
                )
                model = AuxV3Net(config)
            metrics = train_v3_1(
                config, split, run_dir, seed,
                epochs=args.epochs, patience=args.patience,
                batch_size=args.batch_size, grad_accum=args.grad_accum,
                lr=args.lr, encoder_lr=2e-5,
                weight_decay=1e-4, warmup_ratio=0.1, min_lr_ratio=0.1,
                amp=args.amp, device_name=args.device,
                require_cuda=not args.allow_cpu,
                attention_samples=0 if model is not None else 16,
                cache_dir=V3_CACHE, model=model,
                aux_loader=aux_loader,
                aux_weight=args.aux_weight if aux_loader is not None else 0.0,
                aux_source=(f"{aux_source}:n_new={n_new}"
                            if aux_source is not None else None),
            )
            row = _row(name, arch, config, aux_source, n_aux, seed, metrics,
                       args.patience)
            records.append(row)
            _write_summary(out_dir / "metrics", records)
            print(f"[{name}/seed{seed}] test AUC="
                  f"{row['test_roc_auc']:.4f} aux={row['aux_source']} "
                  f"n_aux={n_aux} n_new={n_new}", flush=True)

    stability = _write_summary(out_dir / "metrics", records)
    print("\nSUMMARY", out_dir / "metrics" / "summary.csv")
    for name, val in stability.items():
        print(f"{name:<16} AUC {val['test_roc_auc']['mean']:.4f} "
              f"+/- {val['test_roc_auc']['std']:.4f} "
              f"aux={val['aux_source']} n_aux={val['n_aux']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
