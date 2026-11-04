"""把 V3.1 的 summary.csv 转成报告用的 Markdown 表格。"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

METRICS = ["test_roc_auc", "test_pr_auc", "test_f1", "test_precision",
           "test_recall", "test_mcc", "val_roc_auc", "train_roc_auc",
           "train_test_auc_gap", "best_epoch", "epochs_run",
           "mean_epoch_sec", "peak_allocated_mb", "n_params_total",
           "n_params_trainable"]


def _fmt(v: float) -> str:
    return f"{v:.4f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows = []
    for root in args.roots:
        path = root / "summary.csv"
        if not path.exists():
            continue
        with path.open(encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                rows.append(row)
    groups = defaultdict(list)
    for row in rows:
        groups[row["experiment"]].append(row)

    lines = []
    lines.append("| 实验 | 结构 | n | 测试 AUC | 测试 PR-AUC | 测试 F1 | "
                 "测试 MCC | 验证 AUC | 训练 AUC | 训练-测试 | 最佳 epoch | "
                 "参数量 | 每 epoch 秒 | 峰值显存 MB |")
    lines.append("| --- | --- | ---: | --- | --- | --- | --- | --- | --- | "
                 "--- | --- | ---: | ---: | ---: |")
    for name in sorted(groups):
        items = groups[name]
        def mean_std(key):
            vals = [float(x[key]) for x in items]
            m = sum(vals) / len(vals)
            s = ((sum((v - m) ** 2 for v in vals) / max(len(vals) - 1, 1))
                 ** 0.5)
            return f"{m:.4f} ± {s:.4f}"
        def mean(key):
            return sum(float(x[key]) for x in items) / len(items)
        best = [int(float(x["best_epoch"])) for x in items]
        lines.append(
            f"| `{name}` | {items[0]['arch']} | {len(items)} | "
            f"{mean_std('test_roc_auc')} | {mean_std('test_pr_auc')} | "
            f"{mean_std('test_f1')} | {mean_std('test_mcc')} | "
            f"{mean_std('val_roc_auc')} | {mean_std('train_roc_auc')} | "
            f"{mean_std('train_test_auc_gap')} | "
            f"{'/'.join(str(b) for b in best)} | "
            f"{int(mean('n_params_total')):,} | "
            f"{mean('mean_epoch_sec'):.2f} | "
            f"{mean('peak_allocated_mb'):.0f} |"
        )
    lines.append("")
    lines.append("| 实验 | seed | 验证 AUC | 测试 AUC | PR-AUC | F1 | MCC | "
                 "最佳 epoch | 运行 epoch | 训练 AUC |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
                 "---: | ---: |")
    for name in sorted(groups):
        for r in sorted(groups[name], key=lambda x: int(x["seed"])):
            lines.append(
                f"| `{name}` | {r['seed']} | {_fmt(float(r['val_roc_auc']))} | "
                f"{_fmt(float(r['test_roc_auc']))} | "
                f"{_fmt(float(r['test_pr_auc']))} | "
                f"{_fmt(float(r['test_f1']))} | "
                f"{_fmt(float(r['test_mcc']))} | {r['best_epoch']} | "
                f"{r['epochs_run']} | {_fmt(float(r['train_roc_auc']))} |"
            )
    text = "\n".join(lines)
    print(text)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
