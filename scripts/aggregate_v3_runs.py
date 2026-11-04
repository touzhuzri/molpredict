"""把分散目录中的 V3 run 汇总到统一的 experiments/v3。

用于并行启动多个配置（每个进程写自己的 --out）后合并结果。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path

FIELDS = ["experiment", "modalities", "seed", "val_roc_auc", "val_pr_auc",
          "test_roc_auc", "test_pr_auc", "test_f1", "test_precision",
          "test_recall", "test_mcc", "threshold", "test_tn", "test_fp",
          "test_fn", "test_tp", "best_epoch", "elapsed_sec"]


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    (out / "runs").mkdir(parents=True, exist_ok=True)
    rows, audits, seen_audits = [], [], set()
    for root in args.roots:
        for run in sorted((root / "runs").glob("V3_*")):
            if not (run / "config.json").exists():
                continue
            match = re.match(r"V3_(.+)_seed(\d+)$", run.name)
            if not match:
                raise ValueError(f"无法解析 run 名称: {run.name}")
            name, seed = match.group(1), int(match.group(2))
            cfg = _read_json(run / "config.json")
            met = _read_json(run / "metrics.json")
            t, v = met["test"], met["val"]
            rows.append({
                "experiment": name,
                "modalities": "+".join(cfg["modalities"]),
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
                "best_epoch": cfg.get("best_epoch"),
                "elapsed_sec": cfg.get("elapsed_sec"),
            })
            target = out / "runs" / run.name
            if target.resolve() != run.resolve():
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(run, target)
        audit = root / "leakage_audit.json"
        if audit.exists():
            # 多个并行根目录会各写一份相同划分的审计，按 seed 去重
            for item in _read_json(audit):
                key = item.get("seed", json.dumps(item, sort_keys=True))
                if key in seen_audits:
                    continue
                seen_audits.add(key)
                audits.append(item)
    audits.sort(key=lambda item: item.get("seed", 0))
    rows.sort(key=lambda r: (r["experiment"], r["seed"]))
    with (out / "summary.csv").open("w", encoding="utf-8-sig",
                                    newline="") as f:
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
    (out / "stability.json").write_text(
        json.dumps(stability, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if audits:
        (out / "leakage_audit.json").write_text(
            json.dumps(audits, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(f"aggregated {len(rows)} runs -> {out}")
    for name, val in stability.items():
        print(f"{name:<12} AUC {val['test_roc_auc']['mean']:.4f} "
              f"+/- {val['test_roc_auc']['std']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
