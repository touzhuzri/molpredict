"""分析 SMILES token 长度与测试误差/模型差异的关系。

用途：检查“SMILES 分支是否在小数据上拟合了训练骨架”。
输入是已保存的 V3/V3.1 run 目录；不重新训练、不使用测试集调参。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from molpredict.v3_data import load_cache_arrays, load_rows  # noqa: E402


def _load_predictions(run_dir: Path):
    rows = list(csv.DictReader(
        (run_dir / "predictions_test.csv").open(encoding="utf-8-sig")
    ))
    return {
        "labels": np.asarray([int(r["label"]) for r in rows]),
        "probs": np.asarray([float(r["probability"]) for r in rows]),
        "pred": np.asarray([int(r["predicted"]) for r in rows]),
        "names": [r["name"] for r in rows],
    }


def _safe_auc(y, p):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p)) if len(set(y)) == 2 else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "experiments" / "v3_1" / "smiles_analysis")
    args = parser.parse_args()

    _, labels, rows = load_rows()
    arrays = load_cache_arrays(
        ROOT / "experiments" / "v2" / "cache",
        ROOT / "experiments" / "v3" / "cache",
        [r["inchikey"] for r in rows],
    )
    smiles_len = {r["inchikey"]: len(x) for r, x in zip(rows,
                                                       arrays["smiles_ids"])}
    conf_len = {r["inchikey"]: len(x) for r, x in zip(rows,
                                                     arrays["confseq_ids"])}

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    per_run_samples = {}
    for run_dir in args.runs:
        cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        split = json.loads((run_dir / "split.json").read_text(encoding="utf-8"))
        test_idx = np.asarray(split["test"], dtype=int)
        pred = _load_predictions(run_dir)
        lengths = np.asarray([smiles_len[rows[i]["inchikey"]] for i in test_idx])
        clengths = np.asarray([conf_len[rows[i]["inchikey"]] for i in test_idx])
        err = (pred["pred"] != pred["labels"]).astype(int)
        abs_err = np.abs(pred["probs"] - pred["labels"])
        brier = (pred["probs"] - pred["labels"]) ** 2
        bins = np.quantile(lengths, [0.25, 0.5, 0.75])
        groups = np.digitize(lengths, bins)
        block = {
            "test_auc": _safe_auc(pred["labels"], pred["probs"]),
            "accuracy": float(1 - err.mean()),
            "brier": float(brier.mean()),
            "smiles_length": {
                "mean": float(lengths.mean()),
                "median": float(np.median(lengths)),
                "min": int(lengths.min()), "max": int(lengths.max()),
            },
            "confseq_length": {
                "mean": float(clengths.mean()),
                "median": float(np.median(clengths)),
            },
            "by_smiles_length_quartile": [
                {
                    "quartile": int(q + 1),
                    "n": int((groups == q).sum()),
                    "length_min": int(lengths[groups == q].min()),
                    "length_max": int(lengths[groups == q].max()),
                    "auc": _safe_auc(pred["labels"][groups == q],
                                     pred["probs"][groups == q]),
                    "error_rate": float(err[groups == q].mean()),
                }
                for q in range(4) if (groups == q).any()
            ],
            "spearman_smiles_len_vs_abs_error": float(
                _spearman(lengths, abs_err)
            ),
            "spearman_smiles_len_vs_error": float(
                _spearman(lengths, err)
            ),
        }
        results[run_dir.name] = block
        per_run_samples[run_dir.name] = {
            "idx": test_idx, "err": err, "abs_err": abs_err,
            "brier": brier, "probs": pred["probs"], "labels": pred["labels"],
            "lengths": lengths,
        }

    # 同一 seed 的两个 run 做配对比较（例如 full vs no_smiles）
    paired = {}
    names = list(per_run_samples)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            sa, sb = per_run_samples[a], per_run_samples[b]
            if not np.array_equal(sa["idx"], sb["idx"]):
                continue
            delta = sa["brier"] - sb["brier"]
            long_mask = sa["lengths"] >= np.median(sa["lengths"])
            paired[f"{a}__vs__{b}"] = {
                "same_test_indices": True,
                "mean_brier_delta": float(delta.mean()),
                "mean_brier_delta_long_smiles": float(
                    delta[long_mask].mean()
                ),
                "mean_brier_delta_short_smiles": float(
                    delta[~long_mask].mean()
                ),
                "n_long": int(long_mask.sum()),
                "n_short": int((~long_mask).sum()),
            }

    payload = {"runs": results, "paired": paired,
               "note": "测试集只用于最终诊断，不参与任何模型选择。"}
    path = out_dir / "smiles_effect.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2)[:4000])
    return 0


def _spearman(x, y) -> float:
    from scipy.stats import spearmanr
    if len(set(x)) < 2 or len(set(y)) < 2:
        return 0.0
    return float(spearmanr(x, y).statistic)


if __name__ == "__main__":
    raise SystemExit(main())
