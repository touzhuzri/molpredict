"""用 V3.1 模型预测 21 个外部已知分子（8 杀菌剂 / 13 对照）。

这些分子不在训练集（InChIKey 比对 0 重叠），但属于同一 PPDB 分类体系。
ConfSeq 由官方流程生成，24 个 PaDEL 由 padelpy 计算；不使用替代表示。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from molpredict.v3_predict import V3Predictor  # noqa: E402

CONFIGS = {
    "base": "experiments/v3_1/group1/runs/V3_1_base_seed{seed}",
    "no_smiles": "experiments/v3_1/group3/runs/V3_1_no_smiles_seed{seed}",
    "ft_smiles": "experiments/v3_1/group2/runs/V3_1_ft_smiles_seed{seed}",
    "ft_both": "experiments/v3_1/group2/runs/V3_1_ft_both_seed{seed}",
    "ft_no_smiles": "experiments/v3_1/smiles_check/runs/"
                     "V3_1_ft_no_smiles_seed{seed}",
}


def _metrics(labels, probs):
    return {
        "n": int(len(labels)),
        "n_positive": int(labels.sum()),
        "roc_auc": float(roc_auc_score(labels, probs)),
        "pr_auc": float(average_precision_score(labels, probs)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        default=ROOT / "experiments" / "v3_1" / "external" /
        "external_features.csv",
    )
    parser.add_argument(
        "--out", type=Path,
        default=ROOT / "experiments" / "v3_1" / "external",
    )
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    frame = pd.read_csv(args.input, encoding="utf-8-sig")
    labels = frame["label"].to_numpy(dtype=int)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    probs = {}
    for name, pattern in CONFIGS.items():
        seed_probs = []
        for seed in seeds:
            model_dir = ROOT / pattern.format(seed=seed)
            if not (model_dir / "config.json").exists():
                print(f"skip missing {model_dir}", flush=True)
                continue
            predictor = V3Predictor(model_dir, device=args.device)
            pred = predictor.predict_frame(frame)
            p = pred["probability"].to_numpy(dtype=float)
            seed_probs.append(p)
            probs[f"{name}_seed{seed}"] = p
            print(f"{name} seed{seed}: {p.round(4).tolist()}",
                  flush=True)
        if seed_probs:
            probs[f"{name}_mean"] = np.mean(np.vstack(seed_probs), axis=0)

    result = frame[["group", "name", "smiles", "inchikey", "label"]].copy()
    if "probability" in frame.columns:
        result["catboost_combined"] = frame["probability"]
    elif "catboost_combined" in frame.columns:
        result["catboost_combined"] = frame["catboost_combined"]
    for key, p in probs.items():
        result[key] = p
    result.to_csv(out / "external_predictions.csv", index=False,
                  encoding="utf-8-sig")

    summary = {}
    for key, p in probs.items():
        summary[key] = _metrics(labels, p)
    (out / "external_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
