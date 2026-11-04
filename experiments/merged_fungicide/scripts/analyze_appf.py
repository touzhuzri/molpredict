"""Analyse the APPf-like CatBoost runs.

Produces:
  - aggregated metrics and comparison with the existing deep models;
  - split-42 calibration / applicability-domain analysis;
  - old-only -> new-only source-shift evaluation;
  - label-permutation null test;
  - report figures and tables.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.decomposition import PCA
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, roc_auc_score,
    roc_curve,
)
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from appf_eval import (  # noqa: E402
    bootstrap_ci, ece_score, fit_isotonic, probability_metrics,
    reliability_points, threshold_metrics,
)
from appf_features import (  # noqa: E402
    FEATURE_SETS, build_structure_arrays, load_split, model_matrix,
    tanimoto_to_train, feature_names,
)
from common import DATA, METRICS, OUT, dump_json  # noqa: E402

APPF = OUT / "appf_like"
APPF_METRICS = APPF / "metrics"
PRED = APPF / "predictions"
FIG = APPF / "figures"
FEAT = APPF / "features"
REPORTS = APPF / "reports"
for d in (FIG, FEAT, REPORTS):
    d.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, AQUA, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#7b5aa6"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#c3c2b7")
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def load_all_runs() -> pd.DataFrame:
    frames = []
    for name in ["runs.csv", "runs_OLD_ONLY.csv"]:
        p = APPF_METRICS / name
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        raise FileNotFoundError("no CatBoost run tables")
    df = pd.concat(frames, ignore_index=True)
    return df


def aggregate_runs(df: pd.DataFrame) -> dict:
    out = {}
    for fs, g in df.groupby("feature_set"):
        out[fs] = {
            "n_splits": int(len(g)),
            "test_auc_mean": float(g["test_auc"].mean()),
            "test_auc_std": float(g["test_auc"].std(ddof=1)),
            "test_pr_auc_mean": float(g["test_pr_auc"].mean()),
            "test_pr_auc_std": float(g["test_pr_auc"].std(ddof=1)),
            "test_f1_mean": float(g["test_f1"].mean()),
            "test_f1_std": float(g["test_f1"].std(ddof=1)),
            "test_mcc_mean": float(g["test_mcc"].mean()),
            "test_mcc_std": float(g["test_mcc"].std(ddof=1)),
            "val_auc_mean": float(g["val_auc"].mean()),
            "val_auc_std": float(g["val_auc"].std(ddof=1)),
            "brier_mean": float(g["test_brier"].mean()),
            "ece_mean": float(g["test_ece"].mean()),
        }
    return out


def topk_metrics(y, p, fractions=(0.10, 0.20)) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    order = np.argsort(-p)
    out = {}
    for frac in fractions:
        k = max(1, int(round(len(y) * frac)))
        hit = int(y[order[:k]].sum())
        out[f"top_{int(frac * 100)}pct"] = {
            "k": k, "hits": hit,
            "precision": hit / k,
            "recall": hit / max(int(y.sum()), 1),
        }
    return out


def apply_isotonic_from_val(val_csv: Path, y_val: np.ndarray) -> IsotonicRegression:
    v = pd.read_csv(val_csv)
    iso = fit_isotonic(y_val, v["prob_raw"].to_numpy(float))
    return iso


def evaluate_external(split_seed: int, table: pd.DataFrame,
                      arrays: dict, X: np.ndarray,
                      split: dict, threshold: float) -> tuple[dict, pd.DataFrame]:
    old_keep = ((table["old_positive"] == 1)
                | (table["old_negative"] == 1)).to_numpy()
    train_old = split["train"][old_keep[split["train"]]]
    new_only = ((table["new_positive"] == 1)
                & (table["old_positive"] == 0)
                & (table["old_negative"] == 0)).to_numpy()
    neg_hold = ((table["old_negative"] == 1)
                & (~np.isin(np.arange(len(table)), train_old)))
    ext_idx = np.where(new_only | neg_hold)[0]
    y_ext = table["label"].to_numpy(int)[ext_idx]
    model_dir = APPF / "models" / "EXT_OLD_ONLY" / f"split{split_seed}"
    models = sorted(model_dir.glob("bag*.cbm"))
    if not models:
        raise FileNotFoundError(f"no old-only models in {model_dir}")
    preds = []
    for path in models:
        model = CatBoostClassifier()
        model.load_model(str(path))
        preds.append(model.predict_proba(X[ext_idx])[:, 1])
    p_raw = np.mean(np.vstack(preds), axis=0)
    # Refit the validation isotonic map used for the formal threshold.
    val_csv = PRED / f"EXT_OLD_ONLY_split{split_seed}_val.csv"
    vv = pd.read_csv(val_csv)
    iso = fit_isotonic(vv["label"].to_numpy(int),
                       vv["prob_raw"].to_numpy(float))
    p_cal = iso.predict(p_raw)
    metrics = threshold_metrics(y_ext, p_cal, threshold)
    metrics.update(probability_metrics(y_ext, p_cal))
    metrics["roc_auc_raw"] = float(roc_auc_score(y_ext, p_raw))
    metrics["pr_auc_raw"] = float(average_precision_score(y_ext, p_raw))
    metrics["n_positive"] = int(y_ext.sum())
    metrics["n_negative"] = int((1 - y_ext).sum())
    metrics["topk"] = topk_metrics(y_ext, p_cal)
    df = table.iloc[ext_idx][[
        "structure_id", "canonical_smiles", "label",
        "old_positive", "old_negative", "new_positive",
    ]].copy()
    df["prob_raw"] = p_raw
    df["prob_calibrated"] = p_cal
    df["split_seed"] = split_seed
    return metrics, df


def permutation_test(table: pd.DataFrame, X: np.ndarray,
                     split: dict, n_perm: int = 20) -> dict:
    y = table["label"].to_numpy(int)
    train_idx, test_idx = split["train"], split["test"]
    # Fixed iteration count: no early stopping on the null or observed run.
    kwargs = dict(
        iterations=600, learning_rate=0.03, depth=6, l2_leaf_reg=3.0,
        loss_function="Logloss", random_seed=20261104, verbose=False,
        thread_count=8, allow_writing_files=False,
    )
    obs_model = CatBoostClassifier(**kwargs)
    obs_model.fit(X[train_idx], y[train_idx])
    obs = float(roc_auc_score(y[test_idx],
                              obs_model.predict_proba(X[test_idx])[:, 1]))
    rng = np.random.default_rng(20261104)
    null = []
    for _ in range(n_perm):
        yp = y[train_idx].copy()
        rng.shuffle(yp)
        model = CatBoostClassifier(**kwargs)
        model.fit(X[train_idx], yp)
        null.append(float(roc_auc_score(
            y[test_idx], model.predict_proba(X[test_idx])[:, 1]
        )))
    null = np.asarray(null)
    p_value = (1 + int((null >= obs).sum())) / (1 + len(null))
    return {
        "observed_auc_fixed_600": obs,
        "null_auc_mean": float(null.mean()),
        "null_auc_std": float(null.std(ddof=1)),
        "null_auc_min": float(null.min()),
        "null_auc_max": float(null.max()),
        "n_permutations": int(n_perm),
        "p_value": float(p_value),
        "null_aucs": null.tolist(),
        "note": "Labels shuffled in the training split only; test labels fixed.",
    }


def main() -> int:
    runs = load_all_runs()
    agg = aggregate_runs(runs)
    dump_json(APPF_METRICS / "summary_catboost.json", agg)

    # Existing deep-model results for comparison (split 42, 3 training seeds).
    deep = pd.read_csv(METRICS / "runs.csv")
    deep = deep[(deep["split_seed"] == 42)
                & (deep["experiment"].isin(["A", "B", "C"]))]
    deep_summary = {}
    for exp, g in deep.groupby("experiment"):
        deep_summary[exp] = {
            "test_auc_mean": float(g["test_roc_auc"].mean()),
            "test_auc_std": float(g["test_roc_auc"].std(ddof=1)),
            "test_pr_auc_mean": float(g["test_pr_auc"].mean()),
            "test_pr_auc_std": float(g["test_pr_auc"].std(ddof=1)),
            "test_f1_mean": float(g["test_f1"].mean()),
            "test_mcc_mean": float(g["test_mcc"].mean()),
        }

    table = pd.read_csv(
        DATA / "modeling_structures_seed42.csv", low_memory=False
    ).sort_values("model_index").reset_index(drop=True)
    structures = pd.read_csv(
        DATA / "structures.csv", low_memory=False
    ).sort_values("structure_index").reset_index(drop=True)
    arrays = build_structure_arrays(structures)
    split42 = load_split(42, table)
    X_ext = model_matrix(table, arrays, "EXT")
    y42 = table["label"].to_numpy(int)

    # Calibration / bootstrap / top-k for the selected split-42 test run.
    test_csv = PRED / "EXT_split42_test.csv"
    val_csv = PRED / "EXT_split42_val.csv"
    test_df = pd.read_csv(test_csv)
    val_df = pd.read_csv(val_csv)
    p_test = test_df["prob_calibrated"].to_numpy(float)
    y_test = test_df["label"].to_numpy(int)
    threshold = float(json.loads(
        (APPF_METRICS / "EXT_split42.json").read_text(encoding="utf-8")
    )["threshold"])
    test_metrics = threshold_metrics(y_test, p_test, threshold)
    test_metrics.update(probability_metrics(y_test, p_test))
    test_metrics["bootstrap"] = bootstrap_ci(
        y_test, p_test, threshold, n_boot=2000
    )
    test_metrics["topk"] = topk_metrics(y_test, p_test)
    dump_json(APPF_METRICS / "EXT_split42_selected.json", {
        "threshold": threshold,
        "metrics": test_metrics,
        "note": "Split 42, 25-model EXT ensemble, calibrated on validation.",
    })

    # Applicability domain (Morgan max Tanimoto to training structures).
    tr_idx, va_idx, te_idx = (
        split42["train"], split42["val"], split42["test"]
    )
    train_fp = arrays["morgan"][
        table.iloc[tr_idx]["structure_index"].to_numpy(int)
    ]
    val_fp = arrays["morgan"][
        table.iloc[va_idx]["structure_index"].to_numpy(int)
    ]
    test_fp = arrays["morgan"][
        table.iloc[te_idx]["structure_index"].to_numpy(int)
    ]
    val_max = tanimoto_to_train(val_fp, train_fp)
    test_max = tanimoto_to_train(test_fp, train_fp)
    # Two thresholds:
    #   - a data-driven broad threshold (5th percentile of validation max);
    #   - a conventional strict flag at Tanimoto 0.30, fixed before inspection
    #     of test performance.  The demo app uses the strict flag.
    ad_threshold = float(np.percentile(val_max, 5))
    strict_threshold = 0.30
    in_domain = test_max >= ad_threshold
    strict_in = test_max >= strict_threshold
    ad = {
        "split_seed": 42,
        "threshold": ad_threshold,
        "threshold_rule": "5th percentile of validation max Tanimoto to train",
        "strict_threshold": strict_threshold,
        "strict_rule": "conventional Tanimoto >= 0.30 applicability flag",
        "validation_max_mean": float(val_max.mean()),
        "test_max_mean": float(test_max.mean()),
        "test_in_domain_fraction": float(in_domain.mean()),
        "test_in_domain_n": int(in_domain.sum()),
        "test_out_domain_n": int((~in_domain).sum()),
        "test_strict_in_domain_fraction": float(strict_in.mean()),
        "test_strict_in_domain_n": int(strict_in.sum()),
        "test_strict_out_domain_n": int((~strict_in).sum()),
    }
    if len(np.unique(y_test[in_domain])) > 1:
        ad["in_domain_auc"] = float(roc_auc_score(
            y_test[in_domain], p_test[in_domain]
        ))
    if (~in_domain).sum() > 0 and len(np.unique(y_test[~in_domain])) > 1:
        ad["out_domain_auc"] = float(roc_auc_score(
            y_test[~in_domain], p_test[~in_domain]
        ))
    if len(np.unique(y_test[strict_in])) > 1:
        ad["strict_in_domain_auc"] = float(roc_auc_score(
            y_test[strict_in], p_test[strict_in]
        ))
    if (~strict_in).sum() > 0 and len(np.unique(y_test[~strict_in])) > 1:
        ad["strict_out_domain_auc"] = float(roc_auc_score(
            y_test[~strict_in], p_test[~strict_in]
        ))
    test_df["max_train_tanimoto"] = test_max
    test_df["in_domain"] = in_domain.astype(int)
    test_df["in_domain_strict"] = strict_in.astype(int)
    test_df.to_csv(PRED / "EXT_split42_test_with_ad.csv", index=False)
    dump_json(APPF_METRICS / "applicability_domain_split42.json", ad)

    # Training-fit reference for the selected ensemble.  This is not used for
    # model selection; it only quantifies the train/test gap.
    train_preds = []
    for path in sorted((APPF / "models" / "EXT" / "split42").glob("bag*.cbm")):
        model = CatBoostClassifier()
        model.load_model(str(path))
        train_preds.append(model.predict_proba(X_ext[tr_idx])[:, 1])
    p_train = np.mean(np.vstack(train_preds), axis=0)
    overfit = {
        "train_auc": float(roc_auc_score(y42[tr_idx], p_train)),
        "val_auc": float(roc_auc_score(y42[va_idx],
                                       pd.read_csv(val_csv)["prob_calibrated"]
                                       .to_numpy(float))),
        "test_auc": float(roc_auc_score(y_test, p_test)),
        "note": "Train predictions use the same 25 balanced models; not used "
                "for selection.",
    }
    overfit["train_minus_test"] = overfit["train_auc"] - overfit["test_auc"]
    dump_json(APPF_METRICS / "overfitting_split42.json", overfit)

    np.save(FEAT / "train_morgan_split42.npy", train_fp)
    (FEAT / "train_structure_ids_split42.json").write_text(
        json.dumps(table.iloc[tr_idx]["structure_id"].tolist(),
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (FEAT / "ad_threshold.json").write_text(
        json.dumps({
            "threshold": ad_threshold,
            "rule": ad["threshold_rule"],
            "strict_threshold": strict_threshold,
            "strict_rule": ad["strict_rule"],
        }, indent=2),
        encoding="utf-8",
    )

    # Source-shift evaluation: old-only training -> new-only positives.
    ext_all = {}
    ext_frames = []
    for split_seed in [42, 43, 44]:
        tbl = pd.read_csv(
            DATA / f"modeling_structures_seed{split_seed}.csv",
            low_memory=False,
        ).sort_values("model_index").reset_index(drop=True)
        spl = load_split(split_seed, tbl)
        X = model_matrix(tbl, arrays, "EXT")
        meta = json.loads(
            (APPF_METRICS / f"EXT_OLD_ONLY_split{split_seed}.json").read_text(
                encoding="utf-8"
            )
        )
        m, frame = evaluate_external(
            split_seed, tbl, arrays, X, spl, float(meta["threshold"])
        )
        ext_all[str(split_seed)] = m
        ext_frames.append(frame)
    ext_df = pd.concat(ext_frames, ignore_index=True)
    ext_df.to_csv(PRED / "EXT_OLD_ONLY_external_all_splits.csv", index=False)
    ext_summary = {}
    for k, m in ext_all.items():
        ext_summary[k] = {
            "roc_auc": m.get("roc_auc"), "pr_auc": m.get("pr_auc"),
            "f1": m.get("f1"), "precision": m.get("precision"),
            "recall": m.get("recall"), "mcc": m.get("mcc"),
            "n_positive": m.get("n_positive"),
            "n_negative": m.get("n_negative"),
            "top_10pct": m.get("topk", {}).get("top_10pct"),
            "top_20pct": m.get("topk", {}).get("top_20pct"),
        }
    ext_aucs = [v["roc_auc"] for v in ext_summary.values()
                if v["roc_auc"] is not None]
    ext_summary["mean"] = {
        "roc_auc_mean": float(np.mean(ext_aucs)),
        "roc_auc_std": float(np.std(ext_aucs, ddof=1)),
    }
    dump_json(APPF_METRICS / "source_shift_old_only.json", ext_summary)

    # Permutation null on split 42.
    perm = permutation_test(table, X_ext, split42, n_perm=50)
    dump_json(APPF_METRICS / "permutation_test_split42.json", perm)

    # ---------------- figures ----------------
    # 1. Model comparison.
    labels, means, stds, colors = [], [], [], []
    for name, src, color in [
        ("P24\nPaDEL only", agg.get("P24"), BLUE),
        ("EXT\nPaDEL+FP", agg.get("EXT"), ORANGE),
        ("EXT old-only\ntraining", agg.get("EXT_OLD_ONLY"), AQUA),
        ("Deep C\n(SMILES+PaDEL)", deep_summary.get("C"), PURPLE),
    ]:
        if not src:
            continue
        labels.append(name)
        means.append(src["test_auc_mean"])
        stds.append(src["test_auc_std"])
        colors.append(color)
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    xs = np.arange(len(labels))
    ax.bar(xs, means, width=0.55, color=colors, edgecolor="white", linewidth=2)
    ax.errorbar(xs, means, yerr=stds, fmt="none", ecolor="#52514e",
                elinewidth=1.4, capsize=4)
    for x, m, s in zip(xs, means, stds):
        ax.text(x, m + s + 0.012, f"{m:.3f}", ha="center", va="bottom",
                fontsize=9, color=INK)
    ax.set_xticks(xs, labels, fontsize=8.5)
    ax.set_ylim(0.50, 1.0)
    ax.set_ylabel("Test ROC-AUC", fontsize=9, color=MUTED)
    ax.set_title("APPf-like CatBoost vs existing deep models",
                 fontsize=11.5, color=INK, loc="left")
    style(ax)
    fig.text(0.02, 0.01,
             "CatBoost bars are mean ± SD over 3 scaffold splits; deep bars "
             "are 3 seeds on split 42.  Test is not used for selection.",
             fontsize=7.8, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIG / "model_comparison.png", dpi=200)
    plt.close(fig)

    # 2. ROC and PR for the selected split-42 EXT model.
    fpr, tpr, _ = roc_curve(y_test, p_test)
    prec, rec, _ = precision_recall_curve(y_test, p_test)
    auc = roc_auc_score(y_test, p_test)
    ap = average_precision_score(y_test, p_test)
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.9))
    axes[0].plot(fpr, tpr, color=ORANGE, linewidth=2)
    axes[0].plot([0, 1], [0, 1], color="#b8b7ae", linewidth=1, linestyle="--")
    axes[0].set_xlabel("False positive rate", fontsize=9, color=MUTED)
    axes[0].set_ylabel("True positive rate", fontsize=9, color=MUTED)
    axes[0].set_title(f"ROC-AUC = {auc:.3f}", fontsize=10.5, color=INK,
                      loc="left")
    axes[1].plot(rec, prec, color=ORANGE, linewidth=2)
    axes[1].set_xlabel("Recall", fontsize=9, color=MUTED)
    axes[1].set_ylabel("Precision", fontsize=9, color=MUTED)
    axes[1].set_title(f"PR-AUC = {ap:.3f}", fontsize=10.5, color=INK,
                      loc="left")
    for ax in axes:
        style(ax)
    fig.suptitle("EXT CatBoost, scaffold test split 42", fontsize=11.5,
                 color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / "roc_pr_ext.png", dpi=200)
    plt.close(fig)

    # 3. Calibration.
    rel = reliability_points(y_test, p_test, n_bins=10)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.plot([0, 1], [0, 1], color="#b8b7ae", linestyle="--", linewidth=1)
    ax.plot([r["mean_predicted"] for r in rel],
            [r["fraction_positive"] for r in rel],
            marker="o", markersize=5, color=ORANGE, linewidth=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted score", fontsize=9, color=MUTED)
    ax.set_ylabel("Observed positive fraction", fontsize=9, color=MUTED)
    ax.set_title(f"Calibration (ECE = {ece_score(y_test, p_test):.3f})",
                 fontsize=10.5, color=INK, loc="left")
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "calibration_ext.png", dpi=200)
    plt.close(fig)

    # 4. Score distribution.
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    bins = np.linspace(0, 1, 31)
    ax.hist(p_test[y_test == 0], bins=bins, color=BLUE, alpha=0.72,
            label="Negative (PPDB category)", edgecolor="white", linewidth=0.8)
    ax.hist(p_test[y_test == 1], bins=bins, color=ORANGE, alpha=0.72,
            label="Positive (documented fungicide use)", edgecolor="white",
            linewidth=0.8)
    ax.axvline(threshold, color=INK, linestyle="--", linewidth=1.2,
               label=f"validation threshold = {threshold:.3f}")
    ax.set_xlabel("Calibrated priority score", fontsize=9, color=MUTED)
    ax.set_ylabel("Structures", fontsize=9, color=MUTED)
    ax.set_title("Test score distribution", fontsize=10.5, color=INK,
                 loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "score_distribution.png", dpi=200)
    plt.close(fig)

    # 5. Applicability domain.
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    for lab, color, name in [(0, BLUE, "Negative"), (1, ORANGE, "Positive")]:
        mask = y_test == lab
        ax.scatter(test_max[mask], p_test[mask], s=24, alpha=0.72,
                   color=color, edgecolor="white", linewidth=0.5, label=name)
    ax.axvline(strict_threshold, color=INK, linestyle="--", linewidth=1.1,
               label=f"AD flag = {strict_threshold:.2f}")
    ax.axhline(threshold, color="#b8b7ae", linestyle=":", linewidth=1.0)
    ax.set_xlabel("Max Tanimoto similarity to training structures",
                  fontsize=9, color=MUTED)
    ax.set_ylabel("Calibrated priority score", fontsize=9, color=MUTED)
    ax.set_ylim(0, 1.02)
    ax.set_title("Applicability domain on the test split", fontsize=10.5,
                 color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "applicability_domain.png", dpi=200)
    plt.close(fig)

    # 6. Permutation null.
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.hist(perm["null_aucs"], bins=12, color=BLUE, alpha=0.75,
            edgecolor="white", linewidth=0.8)
    ax.axvline(perm["observed_auc_fixed_600"], color=ORANGE, linewidth=2,
               label=f"observed = {perm['observed_auc_fixed_600']:.3f}")
    ax.axvline(0.5, color="#b8b7ae", linestyle="--", linewidth=1,
               label="chance = 0.500")
    ax.set_xlabel("Test ROC-AUC under shuffled training labels", fontsize=9,
                  color=MUTED)
    ax.set_ylabel("Count", fontsize=9, color=MUTED)
    ax.set_title(f"Label-permutation null (p = {perm['p_value']:.3f})",
                 fontsize=10.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "permutation_null.png", dpi=200)
    plt.close(fig)

    # 7. Source shift.
    keys = ["42", "43", "44"]
    vals = [ext_summary[k]["roc_auc"] for k in keys]
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    xs = np.arange(len(keys))
    ax.bar(xs, vals, width=0.5, color=AQUA, edgecolor="white", linewidth=2)
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.012, f"{v:.3f}", ha="center", va="bottom",
                fontsize=9, color=INK)
    ax.axhline(0.5, color="#b8b7ae", linestyle="--", linewidth=1)
    ax.set_xticks(xs, [f"split {k}" for k in keys])
    ax.set_ylim(0.50, 1.0)
    ax.set_ylabel("ROC-AUC", fontsize=9, color=MUTED)
    ax.set_title("Old-only training -> new-only positives (source shift)",
                 fontsize=10.5, color=INK, loc="left")
    style(ax)
    fig.text(0.02, 0.01,
             "Positives: all 293 structures with only new-source positive "
             "evidence.\nNegatives: held-out old PPDB negatives. "
             "This is a source-shift check, not a true external assay set.",
             fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(FIG / "source_shift.png", dpi=200)
    plt.close(fig)

    # 8. Chemical space PCA.
    pca = PCA(n_components=2, random_state=42)
    model_morgan = arrays["morgan"][
        table["structure_index"].to_numpy(int)
    ].astype(float)
    coords = pca.fit_transform(model_morgan)
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    split_colors = {"train": BLUE, "val": ORANGE, "test": AQUA}
    for name, idx in [("train", tr_idx), ("val", va_idx), ("test", te_idx)]:
        ax.scatter(coords[idx, 0], coords[idx, 1], s=10, alpha=0.65,
                   color=split_colors[name], label=name, linewidth=0)
    ax.set_xlabel("PC1", fontsize=9, color=MUTED)
    ax.set_ylabel("PC2", fontsize=9, color=MUTED)
    ax.set_title("Morgan chemical space by scaffold split", fontsize=10.5,
                 color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, markerscale=2)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "chemical_space.png", dpi=200)
    plt.close(fig)

    # Combined summary.
    summary = {
        "catboost": agg,
        "deep_models_split42": deep_summary,
        "selected_demo_model": "EXT",
        "selected_reason": "highest validation ROC-AUC among the pre-specified "
                           "feature sets; test evaluated once",
        "split42_selected": {
            "threshold": threshold,
            "metrics": test_metrics,
        },
        "applicability_domain": ad,
        "overfitting_split42": overfit,
        "source_shift_old_only": ext_summary,
        "permutation_test": perm,
        "figures": sorted(p.name for p in FIG.glob("*.png")),
    }
    dump_json(APPF_METRICS / "summary.json", summary)

    # Markdown table for the report.
    lines = [
        "# APPf-like results",
        "",
        "## CatBoost across scaffold splits",
        "",
        "| feature set | test AUC | test PR-AUC | test F1 | test MCC | val AUC |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, m in agg.items():
        lines.append(
            f"| {name} | {m['test_auc_mean']:.4f} ± {m['test_auc_std']:.4f} "
            f"| {m['test_pr_auc_mean']:.4f} ± {m['test_pr_auc_std']:.4f} "
            f"| {m['test_f1_mean']:.4f} ± {m['test_f1_std']:.4f} "
            f"| {m['test_mcc_mean']:.4f} ± {m['test_mcc_std']:.4f} "
            f"| {m['val_auc_mean']:.4f} |"
        )
    lines += ["", "## Split-42 selected EXT model", "",
              "| metric | value |", "| --- | ---: |",
              f"| threshold | {threshold:.4f} |",
              f"| ROC-AUC | {test_metrics['roc_auc']:.4f} |",
              f"| PR-AUC | {test_metrics['pr_auc']:.4f} |",
              f"| F1 | {test_metrics['f1']:.4f} |",
              f"| precision | {test_metrics['precision']:.4f} |",
              f"| recall | {test_metrics['recall']:.4f} |",
              f"| MCC | {test_metrics['mcc']:.4f} |",
              f"| Brier | {test_metrics['brier']:.4f} |",
              f"| ECE (10 bins) | {test_metrics['ece_10']:.4f} |"]
    lines += ["", "## Source shift (old-only -> new-only positives)", "",
              "| split | ROC-AUC | PR-AUC | F1 | MCC | top-10% precision |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for k in keys:
        m = ext_summary[k]
        lines.append(
            f"| {k} | {m['roc_auc']:.4f} | {m['pr_auc']:.4f} "
            f"| {m['f1']:.4f} | {m['mcc']:.4f} "
            f"| {m['top_10pct']['precision']:.4f} |"
        )
    lines += ["", "## Train/validation/test fit (split 42)", "",
              f"- train AUC: {overfit['train_auc']:.4f}",
              f"- validation AUC: {overfit['val_auc']:.4f}",
              f"- test AUC: {overfit['test_auc']:.4f}",
              f"- train minus test: {overfit['train_minus_test']:.4f}",
              "", "## Label permutation", "",
              f"- observed fixed-iteration AUC: "
              f"{perm['observed_auc_fixed_600']:.4f}",
              f"- null mean: {perm['null_auc_mean']:.4f} ± "
              f"{perm['null_auc_std']:.4f}",
              f"- empirical p: {perm['p_value']:.4f} (n={perm['n_permutations']})"]
    (REPORTS / "results_tables.md").write_text("\n".join(lines) + "\n",
                                               encoding="utf-8")
    print(json.dumps({
        "catboost": agg,
        "selected": test_metrics,
        "source_shift": ext_summary,
        "permutation": {k: v for k, v in perm.items() if k != "null_aucs"},
        "ad": ad,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
