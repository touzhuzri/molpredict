"""Evaluation helpers for the APPf-like models."""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, log_loss, matthews_corrcoef, precision_score, recall_score,
    roc_auc_score,
)

from common import binary_metrics, best_threshold


def ece_score(y_true, proba, n_bins: int = 10) -> float:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(proba, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if not np.any(mask):
            continue
        total += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(total)


def reliability_points(y_true, proba, n_bins: int = 10) -> list[dict]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(proba, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if not np.any(mask):
            continue
        out.append({
            "bin_low": float(lo), "bin_high": float(hi),
            "n": int(mask.sum()), "mean_predicted": float(p[mask].mean()),
            "fraction_positive": float(y[mask].mean()),
        })
    return out


def fit_isotonic(y_val, p_val) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(np.asarray(p_val, dtype=float), np.asarray(y_val, dtype=int))
    return iso


def probability_metrics(y_true, proba) -> dict:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(proba, dtype=float)
    out = {
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "ece_10": ece_score(y, p, n_bins=10),
        "reliability": reliability_points(y, p, n_bins=10),
    }
    return out


def bootstrap_ci(y_true, proba, threshold: float, n_boot: int = 2000,
                 seed: int = 20261104) -> dict:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(proba, dtype=float)
    if len(np.unique(y)) < 2:
        return {}
    rng = np.random.default_rng(seed)
    aucs, prs, f1s = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        yy, pp = y[idx], p[idx]
        if len(np.unique(yy)) < 2:
            continue
        aucs.append(float(roc_auc_score(yy, pp)))
        prs.append(float(average_precision_score(yy, pp)))
        f1s.append(float(f1_score(yy, (pp >= threshold).astype(int),
                                  zero_division=0)))
    def ci(v):
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {
        "n_boot": int(len(aucs)),
        "roc_auc_95ci": ci(aucs), "pr_auc_95ci": ci(prs),
        "f1_95ci": ci(f1s),
    }


def full_metrics(y_true, proba, threshold: float | None = None,
                 calibrate: bool = True) -> dict:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(proba, dtype=float)
    if threshold is None:
        threshold = best_threshold(y, p)
    base = binary_metrics(y, p, float(threshold))
    base.update(probability_metrics(y, p))
    base["bootstrap"] = bootstrap_ci(y, p, float(threshold))
    if calibrate:
        # The caller must pass validation predictions separately if it wants
        # to fit calibration on validation.  Here the same data are used only
        # for an internal sanity check; the training script overrides this.
        iso = fit_isotonic(y, p)
        pc = iso.predict(p)
        base["internal_calibration"] = {
            "brier": float(brier_score_loss(y, pc)),
            "log_loss": float(log_loss(y, pc, labels=[0, 1])),
            "ece_10": ece_score(y, pc, n_bins=10),
        }
    return base


def threshold_metrics(y_true, proba, threshold: float) -> dict:
    return binary_metrics(np.asarray(y_true, dtype=int),
                          np.asarray(proba, dtype=float), float(threshold))


def confusion(y_true, proba, threshold: float) -> dict:
    y = np.asarray(y_true, dtype=int)
    pred = (np.asarray(proba, dtype=float) >= threshold).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    return {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]), "tp": int(cm[1, 1])}


def point_metrics(y_true, proba) -> dict:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(proba, dtype=float)
    return {
        "n": int(len(y)), "positives": int(y.sum()),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None,
        "pr_auc": float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else None,
    }
