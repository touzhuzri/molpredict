"""Aggregate merged fungicide run metrics into tables."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import METRICS, OUT  # noqa: E402

METRIC_KEYS = ["test_roc_auc", "test_pr_auc", "test_f1", "test_mcc",
               "test_precision", "test_recall", "val_roc_auc",
               "train_roc_auc", "train_test_auc_gap"]


def load_runs() -> pd.DataFrame:
    rows = []
    for path in sorted(METRICS.glob("*.json")):
        if path.name in {"stability.json", "padel_permutation.json",
                         "summary.json"}:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if "test" not in data or "experiment" not in data:
            continue
        row = {
            "experiment": data["experiment"],
            "split_seed": data["split_seed"],
            "seed": data["seed"],
            "best_epoch": data["best_epoch"],
            "epochs_run": data["epochs_run"],
            "n_params_total": data["n_params_total"],
            "elapsed_sec": data["elapsed_sec"],
            "mean_epoch_sec": data["mean_epoch_sec"],
            "peak_allocated_mb": data["peak_allocated_mb"],
            "threshold": data["threshold"],
            "n_train": data["n_train"],
            "n_val": data["n_val"],
            "n_test": data["n_test"],
        }
        t, v, tr = data["test"], data["val"], data["train"]
        row.update({
            "test_roc_auc": t["roc_auc"], "test_pr_auc": t["pr_auc"],
            "test_f1": t["f1"], "test_mcc": t["mcc"],
            "test_precision": t["precision"], "test_recall": t["recall"],
            "test_tn": t["confusion_matrix"]["tn"],
            "test_fp": t["confusion_matrix"]["fp"],
            "test_fn": t["confusion_matrix"]["fn"],
            "test_tp": t["confusion_matrix"]["tp"],
            "val_roc_auc": v["roc_auc"], "val_pr_auc": v["pr_auc"],
            "train_roc_auc": tr["roc_auc"],
            "train_pr_auc": tr["pr_auc"],
            "train_test_auc_gap": data["train_test_auc_gap"],
        })
        for subset, m in data.get("test_subsets", {}).items():
            row[f"{subset}_auc"] = m.get("roc_auc")
            row[f"{subset}_pr"] = m.get("pr_auc")
            row[f"{subset}_n"] = m.get("n")
        rows.append(row)
    return pd.DataFrame(rows)


def ms(series):
    vals = [float(x) for x in series if x is not None and not pd.isna(x)]
    if not vals:
        return None, None
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return mean, std


def fmt(mean, std):
    if mean is None:
        return "NA"
    return f"{mean:.4f} ± {std:.4f}"


def paired(df: pd.DataFrame, a: str, b: str, split_seed: int | None = None):
    aa = df[df["experiment"] == a]
    bb = df[df["experiment"] == b]
    if split_seed is not None:
        aa = aa[aa["split_seed"] == split_seed]
        bb = bb[bb["split_seed"] == split_seed]
    keys = ["split_seed", "seed"]
    merged = aa.merge(bb, on=keys, suffixes=("_a", "_b"))
    if merged.empty:
        return None
    out = {}
    for metric in ("test_roc_auc", "test_pr_auc", "test_f1", "test_mcc"):
        d = merged[f"{metric}_a"] - merged[f"{metric}_b"]
        out[metric] = (float(d.mean()),
                       float(d.std(ddof=1)) if len(d) > 1 else 0.0)
    out["n_pairs"] = int(len(merged))
    return out


def main() -> int:
    df = load_runs()
    if df.empty:
        raise SystemExit("no metrics found")
    df = df.sort_values(["experiment", "split_seed", "seed"])
    df.to_csv(METRICS / "runs.csv", index=False, encoding="utf-8-sig")

    # Summary by experiment and by split seed.
    lines = ["# 合并数据实验结果", ""]
    lines += ["## 按实验汇总（所有 split seed 合并）", "",
              "| 实验 | n runs | 测试 AUC | 测试 PR-AUC | 测试 F1 | 测试 MCC | "
              "验证 AUC | 训练 AUC | 训练-测试 |",
              "| --- | ---: | --- | --- | --- | --- | --- | --- | --- |"]
    for exp, g in df.groupby("experiment"):
        vals = []
        for key in ("test_roc_auc", "test_pr_auc", "test_f1", "test_mcc",
                    "val_roc_auc", "train_roc_auc", "train_test_auc_gap"):
            vals.append(fmt(*ms(g[key])))
        lines.append(f"| `{exp}` | {len(g)} | " + " | ".join(vals) + " |")

    lines += ["", "## 每个 scaffold split 的测试 AUC", "",
              "| 实验 | split | n | 测试 AUC | 测试 PR-AUC | 测试 F1 | 测试 MCC |",
              "| --- | ---: | ---: | --- | --- | --- | --- |"]
    for (exp, ss), g in df.groupby(["experiment", "split_seed"]):
        lines.append(
            f"| `{exp}` | {ss} | {len(g)} | "
            f"{fmt(*ms(g['test_roc_auc']))} | {fmt(*ms(g['test_pr_auc']))} | "
            f"{fmt(*ms(g['test_f1']))} | {fmt(*ms(g['test_mcc']))} |"
        )

    lines += ["", "## 逐 run 测试 AUC", "",
              "| 实验 | split | seed | test AUC | PR-AUC | F1 | MCC | "
              "best epoch | epochs | 参数 | 秒/epoch | 峰值显存 MB |",
              "| --- | ---: | ---: | --- | --- | --- | --- | ---: | ---: | "
              "---: | ---: | ---: |"]
    for _, r in df.iterrows():
        lines.append(
            f"| `{r['experiment']}` | {r['split_seed']} | {r['seed']} | "
            f"{r['test_roc_auc']:.4f} | {r['test_pr_auc']:.4f} | "
            f"{r['test_f1']:.4f} | {r['test_mcc']:.4f} | "
            f"{r['best_epoch']} | {r['epochs_run']} | "
            f"{int(r['n_params_total']):,} | {r['mean_epoch_sec']:.2f} | "
            f"{r['peak_allocated_mb']:.0f} |"
        )

    # Paired differences on split seed 42 (the primary split).
    lines += ["", "## 配对差（同一 scaffold split / seed）", ""]
    pairs = [("B", "A"), ("C", "B")]
    if "B_OLD_ONLY" in set(df["experiment"]):
        pairs.append(("B", "B_OLD_ONLY"))
    lines += ["| 对比 | ΔAUC | ΔPR-AUC | ΔF1 | ΔMCC | pairs |",
              "| --- | --- | --- | --- | --- | ---: |"]
    for a, b in pairs:
        p = paired(df, a, b, split_seed=42)
        if p is None:
            continue
        cells = [f"{p[m][0]:+.4f} ± {p[m][1]:.4f}"
                 for m in ("test_roc_auc", "test_pr_auc", "test_f1",
                           "test_mcc")]
        lines.append(f"| `{a}` − `{b}` | " + " | ".join(cells)
                     + f" | {p['n_pairs']} |")

    # Test subset performance for the primary split.
    lines += ["", "## 测试子集（split 42，按实验）", "",
              "| 实验 | 全部 AUC | 旧正样本 vs 负 | 新独有正 vs 负 | "
              "多成分正 vs 负 | 单成分正 vs 负 | 旧结构子集 AUC |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    sub = df[df["split_seed"] == 42]
    for exp, g in sub.groupby("experiment"):
        lines.append(
            f"| `{exp}` | {fmt(*ms(g['test_roc_auc']))} | "
            f"{fmt(*ms(g['old_positive_vs_negative_auc']))} | "
            f"{fmt(*ms(g['new_only_positive_vs_negative_auc']))} | "
            f"{fmt(*ms(g['multi_component_positive_vs_negative_auc']))} | "
            f"{fmt(*ms(g['single_component_positive_vs_negative_auc']))} | "
            f"{fmt(*ms(g['old_structures_only_auc']))} |"
        )

    # Cost table.
    lines += ["", "## 训练成本（split 42）", "",
              "| 实验 | 参数量 | 平均最佳 epoch | 平均总轮数 | 秒/epoch | "
              "总训练秒 | 峰值显存 MB |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for exp, g in sub.groupby("experiment"):
        lines.append(
            f"| `{exp}` | {int(g['n_params_total'].iloc[0]):,} | "
            f"{g['best_epoch'].mean():.1f} | {g['epochs_run'].mean():.1f} | "
            f"{g['mean_epoch_sec'].mean():.2f} | "
            f"{g['elapsed_sec'].mean():.1f} | "
            f"{g['peak_allocated_mb'].mean():.0f} |"
        )

    text = "\n".join(lines)
    (METRICS / "tables.md").write_text(text, encoding="utf-8")
    summary = {}
    for exp, g in df.groupby("experiment"):
        summary[exp] = {
            k: {"mean": ms(g[k])[0], "std": ms(g[k])[1],
                "values": [float(x) for x in g[k].dropna()]}
            for k in METRIC_KEYS
        }
        summary[exp]["n_runs"] = int(len(g))
        summary[exp]["best_epochs"] = g["best_epoch"].astype(int).tolist()
    (METRICS / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
