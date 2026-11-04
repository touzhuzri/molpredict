"""V3.1 实验入口：GPU/AMP、加深融合网络、微调策略与消融。

V3.1 复用 V3 的缓存、scaffold 划分和统一 token 序列结构；只改变训练配方和
融合网络的深度/宽度。所有输出写入独立目录，不覆盖 V3。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from molpredict.v3_1_train import (  # noqa: E402
    code_manifest,
    gpu_info,
    train_v3_1,
)
from molpredict.v3_data import (  # noqa: E402
    V3_CACHE,
    leakage_audit,
    load_rows,
    load_split,
)
from molpredict.v3_model import V3Config  # noqa: E402

ARCHS = {
    # 名字: (d_model, n_layers, n_heads)
    "base": (128, 4, 4),
    "L": (192, 6, 4),
    "M": (256, 6, 8),
    "D": (128, 8, 4),
}

EXPERIMENTS = {
    # 结构族：冻结编码器，使用缓存 token 表示
    "base": {"arch": "base"},
    "L": {"arch": "L"},
    "M": {"arch": "M"},
    "D": {"arch": "D"},
    # 微调策略：结构由 --arch 指定
    "ft_smiles": {"arch": "arg", "smiles_live": True},
    "ft_both": {"arch": "arg", "smiles_live": True, "confseq_live": True},
    # 补充：在最佳微调模型上复查 SMILES 分支（ConfSeq 微调、SMILES 移除）
    "ft_no_smiles": {"arch": "arg",
                      "modalities": ("confseq", "descriptors"),
                      "confseq_live": True},
    # 消融：结构由 --arch 指定
    "no_smiles": {"arch": "arg",
                  "modalities": ("confseq", "descriptors")},
    "no_confseq": {"arch": "arg",
                   "modalities": ("smiles", "descriptors")},
    "no_padel": {"arch": "arg", "modalities": ("smiles", "confseq")},
}

FIELDS = [
    "experiment", "arch", "modalities", "seed",
    "n_params_total", "n_params_trainable", "best_epoch", "patience_used",
    "val_roc_auc", "val_pr_auc", "test_roc_auc", "test_pr_auc",
    "test_f1", "test_precision", "test_recall", "test_mcc",
    "train_roc_auc", "train_pr_auc", "train_test_auc_gap",
    "threshold", "test_tn", "test_fp", "test_fn", "test_tp",
    "epochs_run", "mean_epoch_sec", "peak_allocated_mb", "gpu_name",
    "amp", "device", "elapsed_sec",
]


def _config(name: str, args) -> tuple[V3Config, str]:
    spec = EXPERIMENTS[name]
    arch_name = spec["arch"] if spec["arch"] != "arg" else args.arch
    d_model, n_layers, n_heads = ARCHS[arch_name]
    kwargs = {
        "d_model": d_model, "n_layers": n_layers, "n_heads": n_heads,
        "dropout": args.dropout, "pooling": args.pooling,
    }
    if "modalities" in spec:
        kwargs["modalities"] = spec["modalities"]
    if spec.get("smiles_live"):
        kwargs["smiles_live"] = True
        kwargs["unfreeze_smiles_layers"] = args.unfreeze_smiles
    if spec.get("confseq_live"):
        kwargs["confseq_live"] = True
        kwargs["unfreeze_confseq_layers"] = args.unfreeze_confseq
    return V3Config(**kwargs), arch_name


def _row(name, arch, config, seed, metrics, patience):
    t, v, tr = metrics["test"], metrics["val"], metrics["train"]
    hist = metrics["history"]
    peak = 0.0
    for item in hist:
        peak = max(peak, float(item["gpu"].get("peak_allocated_mb", 0) or 0))
    mean_epoch = (sum(float(h["epoch_sec"]) for h in hist)
                  / max(len(hist), 1))
    return {
        "experiment": name,
        "arch": arch,
        "modalities": "+".join(config.modalities),
        "seed": seed,
        "n_params_total": metrics["n_params_total"],
        "n_params_trainable": metrics["n_params_trainable"],
        "best_epoch": metrics["best_epoch"],
        "patience_used": patience,
        "val_roc_auc": v["roc_auc"], "val_pr_auc": v["pr_auc"],
        "test_roc_auc": t["roc_auc"], "test_pr_auc": t["pr_auc"],
        "test_f1": t["f1"], "test_precision": t["precision"],
        "test_recall": t["recall"], "test_mcc": t["mcc"],
        "train_roc_auc": tr["roc_auc"], "train_pr_auc": tr["pr_auc"],
        "train_test_auc_gap": tr["roc_auc"] - t["roc_auc"],
        "threshold": t["threshold"],
        "test_tn": t["confusion_matrix"]["tn"],
        "test_fp": t["confusion_matrix"]["fp"],
        "test_fn": t["confusion_matrix"]["fn"],
        "test_tp": t["confusion_matrix"]["tp"],
        "epochs_run": len(hist),
        "mean_epoch_sec": round(mean_epoch, 2),
        "peak_allocated_mb": round(peak, 1),
        "gpu_name": gpu_info().get("name", "cpu"),
        "amp": metrics.get("amp", ""),
        "device": metrics.get("device", ""),
        "elapsed_sec": metrics["elapsed_sec"],
    }


def write_summary(out_dir: Path, rows) -> dict:
    with (out_dir / "summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    groups = {}
    for row in rows:
        groups.setdefault(row["experiment"], []).append(row)
    stability = {}
    for name, items in groups.items():
        block = {"n_runs": len(items),
                 "arch": items[0]["arch"],
                 "modalities": items[0]["modalities"]}
        for metric in ("test_roc_auc", "test_pr_auc", "test_f1",
                       "test_precision", "test_recall", "test_mcc",
                       "val_roc_auc", "train_roc_auc",
                       "train_test_auc_gap"):
            vals = [float(x[metric]) for x in items]
            mean = sum(vals) / len(vals)
            std = (sum((v - mean) ** 2 for v in vals)
                   / max(len(vals) - 1, 1)) ** 0.5
            block[metric] = {"mean": mean, "std": std, "values": vals}
        block["best_epochs"] = [int(x["best_epoch"]) for x in items]
        block["epochs_run"] = [int(x["epochs_run"]) for x in items]
        stability[name] = block
    (out_dir / "stability.json").write_text(
        json.dumps(stability, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return stability


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", default="base,L,D")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--arch", choices=sorted(ARCHS), default="base",
                        help="微调/消融实验使用的融合结构")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--pooling", choices=["cls", "cls_mean"],
                        default="cls")
    parser.add_argument("--unfreeze-smiles", type=int, default=2)
    parser.add_argument("--unfreeze-confseq", type=int, default=2)
    parser.add_argument("--amp", choices=["bf16", "fp16", "off"],
                        default="bf16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-cpu", action="store_true",
                        help="允许 CPU（默认正式实验要求 CUDA）")
    parser.add_argument("--attention-samples", type=int, default=64)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "experiments" / "v3_1")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    names = [s.strip() for s in args.experiments.split(",") if s.strip()]
    unknown = [n for n in names if n not in EXPERIMENTS]
    if unknown:
        raise SystemExit(f"未知实验: {unknown}; 可用: {list(EXPERIMENTS)}")

    mols, labels, rows = load_rows()
    (out_dir / "splits").mkdir(parents=True, exist_ok=True)
    audits = []
    for seed in seeds:
        split = load_split(seed)
        (out_dir / "splits" / f"split_seed{seed}.json").write_text(
            json.dumps({k: v.tolist() for k, v in split.items()},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        audits.append(leakage_audit(mols, rows, split, seed))
    (out_dir / "leakage_audit.json").write_text(
        json.dumps(audits, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "environment.json").write_text(
        json.dumps({
            "gpu": gpu_info(),
            "code_sha256": code_manifest(),
            "torch_threads": os.environ.get("MOLPREDICT_THREADS", "0"),
        }, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    rows_out = []
    for name in names:
        config, arch_name = _config(name, args)
        for seed in seeds:
            split = load_split(seed)
            run_dir = out_dir / "runs" / f"V3_1_{name}_seed{seed}"
            metrics = train_v3_1(
                config, split, run_dir, seed,
                epochs=args.epochs, patience=args.patience,
                batch_size=args.batch_size, grad_accum=args.grad_accum,
                lr=args.lr, encoder_lr=args.encoder_lr,
                weight_decay=args.weight_decay,
                warmup_ratio=args.warmup,
                min_lr_ratio=args.min_lr_ratio,
                amp=args.amp, device_name=args.device,
                require_cuda=not args.allow_cpu,
                attention_samples=args.attention_samples,
                cache_dir=V3_CACHE,
            )
            metrics["amp"] = args.amp
            row = _row(name, arch_name, config, seed, metrics,
                       args.patience)
            rows_out.append(row)
            write_summary(out_dir, rows_out)

    stability = write_summary(out_dir, rows_out)
    print("\n汇总:", out_dir / "summary.csv")
    for name, val in stability.items():
        print(f"{name:<12} AUC {val['test_roc_auc']['mean']:.4f} "
              f"+/- {val['test_roc_auc']['std']:.4f} | "
              f"F1 {val['test_f1']['mean']:.4f} "
              f"+/- {val['test_f1']['std']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
