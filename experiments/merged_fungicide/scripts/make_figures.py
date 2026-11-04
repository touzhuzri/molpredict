"""Create the report figures from metrics/runs.csv."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import METRICS, OUT  # noqa: E402

FIG = OUT / "figures"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#c3c2b7")
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(METRICS / "runs.csv")
    primary = df[df["split_seed"] == 42].copy()
    exps = [e for e in ["A", "B", "C"] if e in set(primary["experiment"])]
    colors = [BLUE, ORANGE, AQUA][:len(exps)]

    # Figure 1: AUC and PR-AUC with individual seed dots.
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.0), sharey=True)
    for ax, metric, label in [
        (axes[0], "test_roc_auc", "Test ROC-AUC"),
        (axes[1], "test_pr_auc", "Test PR-AUC"),
    ]:
        means = [primary.loc[primary.experiment == e, metric].mean()
                 for e in exps]
        stds = [primary.loc[primary.experiment == e, metric].std(ddof=1)
                for e in exps]
        xs = np.arange(len(exps))
        bars = ax.bar(xs, means, width=0.55, color=colors, alpha=0.92,
                      edgecolor="white", linewidth=2)
        ax.errorbar(xs, means, yerr=stds, fmt="none", ecolor="#52514e",
                    elinewidth=1.5, capsize=4, capthick=1.5)
        for i, e in enumerate(exps):
            vals = primary.loc[primary.experiment == e, metric].to_numpy()
            jitter = np.linspace(-0.10, 0.10, len(vals))
            ax.scatter(xs[i] + jitter, vals, s=26, color="white",
                       edgecolor=INK, linewidth=0.8, zorder=5)
            ax.text(xs[i], means[i] + stds[i] + 0.018, f"{means[i]:.3f}",
                    ha="center", va="bottom", fontsize=9, color=INK)
        ax.set_xticks(xs, exps)
        ax.set_title(label, fontsize=11, color=INK, pad=8)
        ax.set_ylim(0.50, 0.90)
        _style(ax)
    axes[0].set_ylabel("Score (mean ± SD, 3 seeds)", fontsize=9,
                       color=MUTED)
    fig.suptitle("Merged fungicide task: scaffold test performance",
                 fontsize=12, color=INK, x=0.02, ha="left")
    fig.text(0.02, 0.01,
             "A: SMILES; B: SMILES + 24 PaDEL; C: B + structure→PaDEL aux. "
             "Fixed scaffold split 42; dots are seeds.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    fig.savefig(FIG / "auc_pr_summary.png", dpi=200)
    plt.close(fig)

    # Figure 2: old-positive vs new-only-positive test subsets.
    if {"old_positive_vs_negative_auc",
            "new_only_positive_vs_negative_auc"}.issubset(primary.columns):
        fig, ax = plt.subplots(figsize=(8.2, 4.1))
        xs = np.arange(len(exps))
        width = 0.34
        for j, (metric, label, color) in enumerate([
            ("old_positive_vs_negative_auc", "Old PPDB positives", BLUE),
            ("new_only_positive_vs_negative_auc", "New-only positives", ORANGE),
        ]):
            means = [primary.loc[primary.experiment == e, metric].mean()
                     for e in exps]
            stds = [primary.loc[primary.experiment == e, metric].std(ddof=1)
                    for e in exps]
            pos = xs + (-width / 2 if j == 0 else width / 2)
            ax.bar(pos, means, width=width, color=color, label=label,
                   edgecolor="white", linewidth=2)
            ax.errorbar(pos, means, yerr=stds, fmt="none", ecolor="#52514e",
                        elinewidth=1.4, capsize=3)
            for x, m, s in zip(pos, means, stds):
                ax.text(x, m + s + 0.018, f"{m:.3f}", ha="center",
                        va="bottom", fontsize=8, color=INK)
        ax.set_xticks(xs, exps)
        ax.set_ylim(0.50, 0.95)
        ax.set_ylabel("ROC-AUC vs shared negatives", fontsize=9, color=MUTED)
        ax.set_title("Performance on old vs new-only positive structures",
                     fontsize=11, color=INK, loc="left")
        ax.legend(frameon=False, fontsize=8, loc="upper left",
                  ncol=2, labelcolor=INK)
        _style(ax)
        fig.text(0.02, 0.01,
                 "Test split 42. New-only positive structures are absent from "
                 "old-only training; subset sizes are small.",
                 fontsize=8, color=MUTED)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        fig.savefig(FIG / "source_subset_auc.png", dpi=200)
        plt.close(fig)

    # Figure 3: PaDEL test-time permutation for B, if available.
    perm_path = METRICS / "padel_permutation_B_split42_seed42.json"
    if perm_path.exists():
        p = json.loads(perm_path.read_text(encoding="utf-8"))
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        labels = ["Baseline", "Row shuffle", "Column shuffle"]
        means = [p["baseline_auc"], p["row_permutation"]["mean"],
                 p["column_permutation"]["mean"]]
        stds = [0.0, p["row_permutation"]["std"],
                p["column_permutation"]["std"]]
        colors = [BLUE, ORANGE, AQUA]
        xs = np.arange(3)
        ax.bar(xs, means, width=0.52, color=colors, edgecolor="white",
               linewidth=2)
        ax.errorbar(xs, means, yerr=stds, fmt="none", ecolor="#52514e",
                    elinewidth=1.4, capsize=4)
        for x, m, s in zip(xs, means, stds):
            ax.text(x, m + s + 0.012, f"{m:.4f}", ha="center", va="bottom",
                    fontsize=8.5, color=INK)
        ax.set_xticks(xs, labels)
        ax.set_ylim(0.50, max(means) + 0.10)
        ax.set_ylabel("Test ROC-AUC", fontsize=9, color=MUTED)
        ax.set_title("B: test-time PaDEL permutation", fontsize=11,
                     color=INK, loc="left")
        _style(ax)
        fig.text(0.02, 0.01,
                 "Only test descriptors are permuted; SMILES/model unchanged. "
                 "Tiny drops indicate the model ignores PaDEL.",
                 fontsize=8, color=MUTED)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        fig.savefig(FIG / "padel_permutation.png", dpi=200)
        plt.close(fig)
    print("figures written to", FIG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
