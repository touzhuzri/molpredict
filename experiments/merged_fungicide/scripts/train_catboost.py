"""APPf-style CatBoost experiments on the merged fungicide dataset.

Two feature sets are trained:
  P24 : exactly the 24 PaDEL descriptors used by the original project/APPf.
  EXT : P24 + 12 RDKit descriptors + Morgan(2048) + MACCS(167).

Each model is an ensemble of 25 balanced subsamples (all positives plus an
equal number of negatives).  This mirrors the balanced-subset idea in APPf
while keeping the scaffold split fixed.  Early stopping and threshold
selection use validation data only.  The test split is evaluated once per
feature set/split seed.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from appf_eval import (  # noqa: E402
    confusion, fit_isotonic, probability_metrics, threshold_metrics,
)
from appf_features import (  # noqa: E402
    FEATURE_SETS, balanced_bags, build_structure_arrays, load_split,
    model_matrix, save_feature_spec,
)
from common import (  # noqa: E402
    DATA, OUT, best_threshold, binary_metrics, dump_json,
)

APPF = OUT / "appf_like"
MODELS = APPF / "models"
APPF_METRICS = APPF / "metrics"
PRED = APPF / "predictions"
LOGS = APPF / "logs"
for d in (MODELS, APPF_METRICS, PRED, LOGS):
    d.mkdir(parents=True, exist_ok=True)


def subset_metrics(table: pd.DataFrame, idx: np.ndarray, y: np.ndarray,
                   proba: np.ndarray, threshold: float) -> dict:
    idx = np.asarray(idx, dtype=int)
    out = {}
    masks = {
        "all": np.ones(len(idx), dtype=bool),
        "old_positive_vs_negative": (
            (table.iloc[idx]["old_positive"].to_numpy() == 1)
            | (table.iloc[idx]["old_negative"].to_numpy() == 1)
        ),
        "new_only_positive_vs_negative": (
            (table.iloc[idx]["new_positive"].to_numpy() == 1)
            & (table.iloc[idx]["old_positive"].to_numpy() == 0)
            & (table.iloc[idx]["old_negative"].to_numpy() == 0)
        ) | (table.iloc[idx]["old_negative"].to_numpy() == 1),
        "multi_component_positive_vs_negative": (
            (table.iloc[idx]["is_multi_component"].to_numpy() == 1)
            & (y == 1)
        ) | (y == 0),
        "single_component_positive_vs_negative": (
            (table.iloc[idx]["is_multi_component"].to_numpy() == 0)
            & (y == 1)
        ) | (y == 0),
        "old_structures_only": (
            (table.iloc[idx]["old_positive"].to_numpy() == 1)
            | (table.iloc[idx]["old_negative"].to_numpy() == 1)
        ),
    }
    for name, m in masks.items():
        yy = y[m]
        pp = proba[m]
        if len(yy) == 0 or len(np.unique(yy)) < 2:
            out[name] = {"n": int(len(yy)), "note": "empty or one class"}
            continue
        out[name] = binary_metrics(yy, pp, threshold)
    return out


def pred_frame(table: pd.DataFrame, idx: np.ndarray, raw: np.ndarray,
               calibrated: np.ndarray, split_seed: int,
               bag_std: np.ndarray | None = None) -> pd.DataFrame:
    df = table.iloc[np.asarray(idx, dtype=int)][[
        "structure_id", "canonical_smiles", "scaffold",
        "label", "old_positive", "old_negative",
        "new_positive", "is_multi_component",
    ]].copy()
    df["prob_raw"] = raw
    df["prob_calibrated"] = calibrated
    if bag_std is not None:
        df["bag_std"] = bag_std
    df["split_seed"] = split_seed
    return df


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-seeds", default="42,43,44")
    parser.add_argument("--feature-sets", default="P24,EXT")
    parser.add_argument("--bags", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--l2", type=float, default=3.0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--old-only", action="store_true")
    parser.add_argument("--save-models-split", default="42")
    args = parser.parse_args()

    split_seeds = [int(x) for x in args.split_seeds.split(",") if x]
    save_model_splits = {int(x) for x in str(args.save_models_split).split(",") if x}
    feature_sets = [x.strip() for x in args.feature_sets.split(",")
                    if x.strip()]
    for fs in feature_sets:
        if fs not in FEATURE_SETS:
            raise ValueError(f"unknown feature set: {fs}")
    suffix = "_OLD_ONLY" if args.old_only else ""
    save_feature_spec(APPF / "feature_spec.json")
    dump_json(APPF / "config.json", {
        "experiment": "APPf-like CatBoost",
        "feature_sets": feature_sets,
        "split_seeds": split_seeds,
        "bags": args.bags,
        "iterations": args.iterations,
        "learning_rate": args.learning_rate,
        "depth": args.depth,
        "l2_leaf_reg": args.l2,
        "old_only": bool(args.old_only),
        "test_protocol": "evaluate once per feature set/split; no test-based "
                         "epoch, feature, threshold or model selection",
    })

    structures = pd.read_csv(
        DATA / "structures.csv", low_memory=False
    ).sort_values("structure_index").reset_index(drop=True)
    arrays = build_structure_arrays(structures)

    rows = []
    for split_seed in split_seeds:
        table = pd.read_csv(
            DATA / f"modeling_structures_seed{split_seed}.csv",
            low_memory=False,
        ).sort_values("model_index").reset_index(drop=True)
        split = load_split(split_seed, table)
        y_all = table["label"].to_numpy(dtype=int)
        old_keep = ((table["old_positive"] == 1)
                    | (table["old_negative"] == 1)).to_numpy()
        if args.old_only:
            train_idx = split["train"][old_keep[split["train"]]]
            val_metric_idx = split["val"][old_keep[split["val"]]]
        else:
            train_idx = split["train"]
            val_metric_idx = split["val"]
        test_idx = split["test"]
        if len(train_idx) == 0 or len(val_metric_idx) == 0:
            raise ValueError("empty train/validation after filtering")

        for fs in feature_sets:
            X = model_matrix(table, arrays, fs)
            bags = balanced_bags(
                train_idx, y_all, n_bags=args.bags,
                seed=10_000 + split_seed * 100,
            )
            val_preds, test_preds, bag_records = [], [], []
            for b, bag_idx in enumerate(bags):
                model = CatBoostClassifier(
                    iterations=args.iterations,
                    learning_rate=args.learning_rate,
                    depth=args.depth,
                    l2_leaf_reg=args.l2,
                    loss_function="Logloss",
                    eval_metric="AUC",
                    random_seed=split_seed * 100 + b,
                    verbose=False,
                    early_stopping_rounds=200,
                    thread_count=args.threads,
                    allow_writing_files=False,
                )
                model.fit(
                    X[bag_idx], y_all[bag_idx],
                    eval_set=(X[val_metric_idx], y_all[val_metric_idx]),
                    use_best_model=True,
                )
                pv = model.predict_proba(X[val_metric_idx])[:, 1]
                pt = model.predict_proba(X[test_idx])[:, 1]
                val_preds.append(pv)
                test_preds.append(pt)
                bag_records.append({
                    "feature_set": fs + suffix,
                    "split_seed": split_seed,
                    "bag": b,
                    "best_iteration": int(model.get_best_iteration()),
                    "val_auc": float(roc_auc_score(
                        y_all[val_metric_idx], pv)),
                    "test_auc": float(roc_auc_score(y_all[test_idx], pt)),
                    "n_train": int(len(bag_idx)),
                })
                if split_seed in save_model_splits:
                    model_dir = MODELS / (fs + suffix) / f"split{split_seed}"
                    model_dir.mkdir(parents=True, exist_ok=True)
                    model.save_model(str(model_dir / f"bag{b:02d}.cbm"))

            p_val = np.mean(np.vstack(val_preds), axis=0)
            p_test = np.mean(np.vstack(test_preds), axis=0)
            bag_std = np.std(np.vstack(test_preds), axis=0)
            iso = fit_isotonic(y_all[val_metric_idx], p_val)
            p_val_cal = iso.predict(p_val)
            p_test_cal = iso.predict(p_test)
            threshold = best_threshold(y_all[val_metric_idx], p_val_cal)

            test_metrics = threshold_metrics(
                y_all[test_idx], p_test_cal, threshold
            )
            test_metrics.update(probability_metrics(
                y_all[test_idx], p_test_cal
            ))
            test_metrics["confusion_matrix"] = confusion(
                y_all[test_idx], p_test_cal, threshold
            )
            val_threshold = best_threshold(y_all[val_metric_idx], p_val_cal)
            val_metrics = threshold_metrics(
                y_all[val_metric_idx], p_val_cal, val_threshold
            )
            val_metrics.update(probability_metrics(
                y_all[val_metric_idx], p_val_cal
            ))
            result = {
                "feature_set": fs + suffix,
                "split_seed": split_seed,
                "n_bags": args.bags,
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_metric_idx)),
                "n_test": int(len(test_idx)),
                "n_train_positive": int(y_all[train_idx].sum()),
                "n_train_negative": int(len(train_idx)
                                        - y_all[train_idx].sum()),
                "threshold": float(threshold),
                "validation": val_metrics,
                "test": test_metrics,
                "test_subsets": subset_metrics(
                    table, test_idx, y_all[test_idx], p_test_cal, threshold
                ),
                "bag_std_mean": float(np.mean(bag_std)),
                "bag_std_max": float(np.max(bag_std)),
                "bag_metrics": bag_records,
                "test_raw": {
                    "roc_auc": float(roc_auc_score(y_all[test_idx], p_test)),
                    "pr_auc": float(average_precision_score(
                        y_all[test_idx], p_test)),
                },
            }
            dump_json(
                APPF_METRICS / f"{fs}{suffix}_split{split_seed}.json",
                result,
            )
            pred_frame(table, val_metric_idx, p_val, p_val_cal,
                       split_seed).to_csv(
                PRED / f"{fs}{suffix}_split{split_seed}_val.csv",
                index=False,
            )
            pred_frame(table, test_idx, p_test, p_test_cal,
                       split_seed, bag_std).to_csv(
                PRED / f"{fs}{suffix}_split{split_seed}_test.csv",
                index=False,
            )
            rows.append({
                "feature_set": fs + suffix,
                "split_seed": split_seed,
                "n_bags": args.bags,
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_metric_idx)),
                "n_test": int(len(test_idx)),
                "threshold": float(threshold),
                "test_auc": test_metrics.get("roc_auc"),
                "test_pr_auc": test_metrics.get("pr_auc"),
                "test_f1": test_metrics.get("f1"),
                "test_precision": test_metrics.get("precision"),
                "test_recall": test_metrics.get("recall"),
                "test_mcc": test_metrics.get("mcc"),
                "test_brier": test_metrics.get("brier"),
                "test_ece": test_metrics.get("ece_10"),
                "val_auc": val_metrics.get("roc_auc"),
                "bag_std_mean": float(np.mean(bag_std)),
                "path": str(
                    APPF_METRICS / f"{fs}{suffix}_split{split_seed}.json"
                ),
            })
            print(json.dumps({
                "feature_set": fs + suffix,
                "split_seed": split_seed,
                "val_auc": round(float(val_metrics.get("roc_auc") or 0), 4),
                "test_auc": round(float(test_metrics.get("roc_auc") or 0), 4),
                "test_pr_auc": round(float(test_metrics.get("pr_auc") or 0), 4),
                "f1": round(float(test_metrics.get("f1") or 0), 4),
                "mcc": round(float(test_metrics.get("mcc") or 0), 4),
            }, ensure_ascii=False))

    runs_path = APPF_METRICS / f"runs{suffix}.csv"
    with runs_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("wrote", runs_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
