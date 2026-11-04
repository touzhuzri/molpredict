"""V2 实验入口：缓存特征、生成划分、训练 A–G 并汇总稳定性。"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from molpredict.modeling import load_dataset  # noqa: E402
from molpredict.v2 import (  # noqa: E402
    DEFAULT_CACHE,
    EXPERIMENTS,
    prepare_cache,
    run_catboost,
    run_neural,
    save_splits,
)


def load_splits(out_dir: Path, seeds):
    splits = {}
    for seed in seeds:
        splits[seed] = json.loads(
            (out_dir / "splits" / f"split_seed{seed}.json").read_text(encoding="utf-8")
        )
        for key in ("train", "val", "test"):
            splits[seed][key] = np.array(splits[seed][key])
    return splits


def write_summary(out_dir: Path, rows):
    path = out_dir / "summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["experiment", "fusion", "seed", "val_roc_auc", "val_pr_auc",
              "test_roc_auc", "test_pr_auc", "test_f1", "test_precision",
              "test_recall", "test_mcc", "threshold", "test_tn", "test_fp",
              "test_fn", "test_tp"]
    # 单独运行 --finetune 时，不要把之前 A–G 的汇总覆盖掉
    if path.exists():
        with path.open(encoding="utf-8-sig", newline="") as f:
            previous = list(csv.DictReader(f))
        key = lambda r: (r["experiment"], r["fusion"], str(r["seed"]))
        merged = {key(r): r for r in previous}
        merged.update({key(r): r for r in rows})
        order = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "F-ft": 6, "G": 7}
        rows = sorted(merged.values(),
                      key=lambda r: (order.get(r["experiment"], 99),
                                     r["fusion"], int(r["seed"])))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    groups = {}
    for row in rows:
        key = (row["experiment"], row["fusion"])
        groups.setdefault(key, []).append(row)
    stability = {}
    for (exp, fusion), items in groups.items():
        key = f"{exp}/{fusion}"
        stability[key] = {"n_runs": len(items)}
        for metric in ("test_roc_auc", "test_pr_auc", "test_f1", "test_mcc"):
            vals = [float(x[metric]) for x in items]
            stability[key][metric] = {
                "mean": sum(vals) / len(vals),
                "std": (sum((v - sum(vals) / len(vals)) ** 2 for v in vals)
                        / max(len(vals) - 1, 1)) ** 0.5,
                "values": vals,
            }
    (out_dir / "stability.json").write_text(
        json.dumps(stability, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return stability


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--experiments", default="A,B,C,D,E,F,G")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--fusions", default="attention,gated")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "v2")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    experiments = [e.strip() for e in args.experiments.split(",") if e.strip()]
    fusions = [f.strip() for f in args.fusions.split(",") if f.strip()]
    out_dir = Path(args.out)
    cache_dir = out_dir / "cache"

    if args.prepare:
        prepare_cache(ROOT / "data/training/confseq.csv", cache_dir)
        return 0

    mols, labels, _ = load_dataset(ROOT / "data/training/confseq.csv")
    splits_dir = out_dir / "splits"
    if not (splits_dir / f"split_seed{seeds[0]}.json").exists():
        save_splits(mols, labels, seeds, splits_dir)
    splits = load_splits(out_dir, seeds)

    rows = []
    for exp in experiments:
        if exp == "A":
            for seed in seeds:
                run_dir = out_dir / "runs" / f"A_catboost_seed{seed}"
                metrics = run_catboost(splits[seed], run_dir, seed)
                rows.append(_row("A", "catboost", seed, metrics))
            continue
        spec = EXPERIMENTS[exp]
        use_fusions = fusions if len(spec["modalities"]) > 1 else ["none"]
        for fusion in use_fusions:
            for seed in seeds:
                run_dir = out_dir / "runs" / f"{exp}_{fusion}_seed{seed}"
                metrics = run_neural(
                    exp, spec["modalities"], spec["numeric"], splits[seed],
                    cache_dir, run_dir, fusion, seed,
                    epochs=args.epochs, patience=args.patience,
                )
                rows.append(_row(exp, fusion, seed, metrics))

    if args.finetune:
        seed = seeds[0]
        spec = EXPERIMENTS["F"]
        run_dir = out_dir / "runs" / f"F_attention_ft_seed{seed}"
        metrics = run_neural(
            "F-ft", spec["modalities"], spec["numeric"], splits[seed],
            cache_dir, run_dir, "attention", seed,
            live_smiles=True, unfreeze_layers=2, epochs=12, patience=4,
        )
        rows.append(_row("F-ft", "attention", seed, metrics))

    stability = write_summary(out_dir, rows)
    print(f"\n汇总: {out_dir / 'summary.csv'}")
    for key, val in stability.items():
        print(f"{key:<22} test AUC {val['test_roc_auc']['mean']:.4f} "
              f"± {val['test_roc_auc']['std']:.4f} | "
              f"F1 {val['test_f1']['mean']:.4f} ± {val['test_f1']['std']:.4f}")
    return 0


def _row(exp, fusion, seed, metrics):
    t = metrics["test"]
    v = metrics["val"]
    return {
        "experiment": exp, "fusion": fusion, "seed": seed,
        "val_roc_auc": v["roc_auc"], "val_pr_auc": v["pr_auc"],
        "test_roc_auc": t["roc_auc"], "test_pr_auc": t["pr_auc"],
        "test_f1": t["f1"], "test_precision": t["precision"],
        "test_recall": t["recall"], "test_mcc": t["mcc"],
        "threshold": t["threshold"],
        "test_tn": t["confusion_matrix"]["tn"],
        "test_fp": t["confusion_matrix"]["fp"],
        "test_fn": t["confusion_matrix"]["fn"],
        "test_tp": t["confusion_matrix"]["tp"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
