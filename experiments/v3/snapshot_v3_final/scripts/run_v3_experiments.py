"""V3 实验入口：统一 Transformer、消融、部分解冻、汇总。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from molpredict.v3_data import (  # noqa: E402
    V3_CACHE,
    load_rows,
    load_split,
    leakage_audit,
    prepare_v3_cache,
)
from molpredict.v3_model import V3Config  # noqa: E402
from molpredict.v3_train import train_v3  # noqa: E402

CONFIGS = {
    "full": V3Config(),
    "no_smiles": V3Config(modalities=("confseq", "descriptors")),
    "no_confseq": V3Config(modalities=("smiles", "descriptors")),
    "no_padel": V3Config(modalities=("smiles", "confseq")),
    # 第二阶段：只解冻 ChemBERTa 最后 2 层；ConfSeq 保持冻结缓存
    "full_ft": V3Config(smiles_live=True, unfreeze_smiles_layers=2),
}
FIELDS = ["experiment", "modalities", "seed", "val_roc_auc", "val_pr_auc",
          "test_roc_auc", "test_pr_auc", "test_f1", "test_precision",
          "test_recall", "test_mcc", "threshold", "test_tn", "test_fp",
          "test_fn", "test_tp", "best_epoch", "elapsed_sec"]


def _row(name, config, seed, metrics):
    t, v = metrics["test"], metrics["val"]
    return {
        "experiment": name,
        "modalities": "+".join(config.modalities),
        "seed": seed,
        "val_roc_auc": v["roc_auc"], "val_pr_auc": v["pr_auc"],
        "test_roc_auc": t["roc_auc"], "test_pr_auc": t["pr_auc"],
        "test_f1": t["f1"], "test_precision": t["precision"],
        "test_recall": t["recall"], "test_mcc": t["mcc"],
        "threshold": t["threshold"],
        "test_tn": t["confusion_matrix"]["tn"],
        "test_fp": t["confusion_matrix"]["fp"],
        "test_fn": t["confusion_matrix"]["fn"],
        "test_tp": t["confusion_matrix"]["tp"],
        "best_epoch": metrics["best_epoch"],
        "elapsed_sec": metrics["elapsed_sec"],
    }


def write_summary(out_dir: Path, rows):
    path = out_dir / "summary.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    groups = {}
    for row in rows:
        groups.setdefault(row["experiment"], []).append(row)
    stability = {}
    for name, items in groups.items():
        stability[name] = {"n_runs": len(items)}
        for metric in ("test_roc_auc", "test_pr_auc", "test_f1", "test_mcc"):
            vals = [float(x[metric]) for x in items]
            mean = sum(vals) / len(vals)
            std = (sum((v - mean) ** 2 for v in vals)
                   / max(len(vals) - 1, 1)) ** 0.5
            stability[name][metric] = {
                "mean": mean, "std": std, "values": vals,
            }
    (out_dir / "stability.json").write_text(
        json.dumps(stability, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return stability


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--experiments", default="full,no_smiles,no_confseq,no_padel")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
    parser.add_argument("--dropout", type=float, default=None,
                        help="覆盖配置中的 dropout")
    parser.add_argument("--pooling", choices=["cls", "cls_mean"], default=None,
                        help="覆盖配置中的最终表示池化")
    parser.add_argument("--quick", action="store_true",
                        help="冒烟测试：每侧最多 64 条、2 epochs")
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "v3")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.prepare:
        manifest = prepare_v3_cache()
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    names = [s.strip() for s in args.experiments.split(",") if s.strip()]
    unknown = [n for n in names if n not in CONFIGS]
    if unknown:
        raise SystemExit(f"未知实验: {unknown}; 可用: {list(CONFIGS)}")

    # 固定复用 V2 的 scaffold 划分，并复制到 V3 以便独立审计
    (out_dir / "splits").mkdir(parents=True, exist_ok=True)
    mols, labels, rows = load_rows()
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

    rows_out = []
    for name in names:
        config = CONFIGS[name]
        if args.dropout is not None or args.pooling is not None:
            payload = config.to_dict()
            if args.dropout is not None:
                payload["dropout"] = args.dropout
            if args.pooling is not None:
                payload["pooling"] = args.pooling
            config = V3Config.from_dict(payload)
        for seed in seeds:
            split = load_split(seed)
            run_dir = out_dir / "runs" / f"V3_{name}_seed{seed}"
            metrics = train_v3(
                config, split, V3_CACHE, run_dir, seed,
                epochs=2 if args.quick else args.epochs,
                patience=1 if args.quick else args.patience,
                batch_size=args.batch_size,
                lr=args.lr, encoder_lr=args.encoder_lr,
                quick_limit=64 if args.quick else None,
            )
            rows_out.append(_row(name, config, seed, metrics))
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
