"""把 new_data_comparison 的 summary.csv 转成 Markdown 表与配对差。"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments" / "new_data_comparison"
METRICS = ["test_roc_auc", "test_pr_auc", "test_f1", "test_mcc",
           "val_roc_auc", "train_roc_auc", "train_test_auc_gap"]


def main() -> int:
    rows = list(csv.DictReader(
        (OUT / "metrics" / "summary.csv").open(encoding="utf-8-sig")
    ))
    groups = defaultdict(list)
    for row in rows:
        groups[row["experiment"]].append(row)

    lines = ["| 实验 | 输入/条件 | n | 测试 AUC | 测试 PR-AUC | 测试 F1 | "
             "测试 MCC | 验证 AUC | 训练 AUC | 训练-测试 |",
             "| --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- |"]
    for name in sorted(groups):
        items = groups[name]
        def ms(key):
            vals = [float(x[key]) for x in items]
            mean = sum(vals) / len(vals)
            std = (sum((v - mean) ** 2 for v in vals)
                   / max(len(vals) - 1, 1)) ** 0.5
            return f"{mean:.4f} ± {std:.4f}"
        cond = items[0]["aux_source"] or "no aux"
        lines.append(
            f"| `{name}` | {items[0]['modalities']} / {cond} | {len(items)} | "
            f"{ms('test_roc_auc')} | {ms('test_pr_auc')} | {ms('test_f1')} | "
            f"{ms('test_mcc')} | {ms('val_roc_auc')} | {ms('train_roc_auc')} | "
            f"{ms('train_test_auc_gap')} |"
        )

    lines += ["", "逐 seed 测试 AUC：", "",
              "| 实验 | seed42 | seed43 | seed44 | 参数量 | 最佳 epoch |",
              "| --- | ---: | ---: | ---: | ---: | --- |"]
    for name in sorted(groups):
        by = {int(x["seed"]): x for x in groups[name]}
        vals = [float(by[s]["test_roc_auc"]) for s in (42, 43, 44)
                if s in by]
        best = "/".join(by[s]["best_epoch"] for s in (42, 43, 44) if s in by)
        lines.append(f"| `{name}` | " + " | ".join(f"{v:.4f}" for v in vals)
                     + f" | {int(by[42]['n_params_total']):,} | {best} |")

    # 配对差（同 seed）
    lines += ["", "关键配对差（同 seed 的测试 AUC 差，均值 ± 标准差）：", "",
              "| 对比 | ΔAUC | ΔPR-AUC | ΔF1 | ΔMCC |",
              "| --- | --- | --- | --- | --- |"]
    pairs = [
        ("C_OLD_ONLY", "A_OLD_ONLY"),
        ("C_OLD_ONLY", "B_OLD_ONLY"),
        ("D_OLD_ONLY", "C_OLD_ONLY"),
        ("D_NEW_AUX_MAIN", "D_OLD_ONLY"),
        ("D_NEW_AUX_FULL", "D_OLD_ONLY"),
        ("E_OLD_ONLY", "C_OLD_ONLY"),
    ]
    for a, b in pairs:
        if a not in groups or b not in groups:
            continue
        aa = {int(x["seed"]): x for x in groups[a]}
        bb = {int(x["seed"]): x for x in groups[b]}
        seeds = sorted(set(aa) & set(bb))
        cells = []
        for metric in ("test_roc_auc", "test_pr_auc", "test_f1", "test_mcc"):
            diffs = [float(aa[s][metric]) - float(bb[s][metric]) for s in seeds]
            mean = sum(diffs) / len(diffs)
            std = (sum((d - mean) ** 2 for d in diffs)
                   / max(len(diffs) - 1, 1)) ** 0.5
            cells.append(f"{mean:+.4f} ± {std:.4f}")
        lines.append(f"| `{a}` − `{b}` | " + " | ".join(cells) + " |")

    text = "\n".join(lines)
    path = OUT / "metrics" / "tables.md"
    path.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
