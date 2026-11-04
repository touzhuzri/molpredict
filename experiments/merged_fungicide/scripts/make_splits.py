"""Create structure-grouped scaffold splits for the merged dataset.

Algorithm:
1. every distinct canonical structure belongs to exactly one split;
2. Murcko scaffolds with >=20 structures are forced into train, otherwise a
   single benzene-like scaffold would dominate the test split;
3. the remaining scaffold groups are assigned in seeded random order to
   whichever split is furthest below its 70/15/15 size target.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    DATA, FORCE_TRAIN_MIN_GROUP, SPLIT_FRACTIONS, SPLITS, dump_json,
)


def build_split(struct: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, dict]:
    trainable = struct[struct["model_eligible"] == 1].copy()
    if trainable["label"].isna().any():
        raise ValueError("trainable structure has missing label")
    trainable["label"] = trainable["label"].astype(int)
    trainable["scaffold"] = trainable["scaffold"].fillna(
        "MISSING_SCAFFOLD:" + trainable["structure_id"]
    )
    trainable = trainable.sort_values("structure_id").reset_index(drop=True)
    trainable["model_index"] = np.arange(len(trainable), dtype=int)

    g = trainable.groupby("scaffold").agg(
        n=("label", "size"), pos=("label", "sum")
    ).reset_index()
    forced = set(g.loc[g["n"] >= FORCE_TRAIN_MIN_GROUP, "scaffold"])
    remaining = g.loc[~g["scaffold"].isin(forced)].sample(
        frac=1, random_state=seed
    ).to_dict("records")

    targets = {k: SPLIT_FRACTIONS[k] * len(trainable)
               for k in SPLIT_FRACTIONS}
    assigned = {k: set() for k in targets}
    assigned["train"] |= forced
    counts = {k: np.array([0.0, 0.0]) for k in targets}
    forced_rows = g[g["scaffold"].isin(forced)]
    counts["train"] = np.array([
        float(forced_rows["n"].sum()), float(forced_rows["pos"].sum())
    ])

    for row in remaining:
        # Fill the split with the smallest current fraction first.  Large
        # groups are already in train, so test/val are built from small
        # scaffold groups and do not get a single dominating scaffold.
        fractions = {k: counts[k][0] / max(targets[k], 1.0)
                     for k in targets}
        k = min(fractions, key=fractions.get)
        assigned[k].add(row["scaffold"])
        counts[k] += np.array([row["n"], row["pos"]])

    split = np.empty(len(trainable), dtype=object)
    for k, scaffolds in assigned.items():
        mask = trainable["scaffold"].isin(scaffolds)
        split[mask.to_numpy()] = k
    trainable["split"] = split
    if trainable["split"].isna().any():
        raise RuntimeError("some structures were not assigned")

    sets = {k: set(trainable.loc[trainable["split"] == k, "scaffold"])
            for k in targets}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if sets[a] & sets[b]:
            raise RuntimeError(f"scaffold leakage between {a} and {b}")

    summary = {
        "seed": seed,
        "n_structures": int(len(trainable)),
        "n_positive": int(trainable["label"].sum()),
        "n_negative": int((1 - trainable["label"]).sum()),
        "forced_train_scaffolds": sorted(forced),
        "forced_train_structures": int(forced_rows["n"].sum()),
        "splits": {},
    }
    for k in ("train", "val", "test"):
        part = trainable[trainable["split"] == k]
        summary["splits"][k] = {
            "n": int(len(part)),
            "positive": int(part["label"].sum()),
            "negative": int((1 - part["label"]).sum()),
            "positive_fraction": float(part["label"].mean()),
            "scaffolds": int(part["scaffold"].nunique()),
            "multi_component": int(part["is_multi_component"].sum()),
            "positive_multi_component": int(
                ((part["label"] == 1) & (part["is_multi_component"] == 1)).sum()
            ),
            "old_only_positive": int((
                (part["label"] == 1) & (part["old_positive"] == 1)
                & (part["new_positive"] == 0)
            ).sum()),
            "both_positive": int((
                (part["label"] == 1) & (part["old_positive"] == 1)
                & (part["new_positive"] == 1)
            ).sum()),
            "new_only_positive": int((
                (part["label"] == 1) & (part["old_positive"] == 0)
                & (part["new_positive"] == 1)
            ).sum()),
        }
    return trainable, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="42,43,44")
    args = parser.parse_args()
    struct = pd.read_csv(DATA / "structures.csv", low_memory=False)
    SPLITS.mkdir(parents=True, exist_ok=True)

    all_summary = {}
    for seed in [int(x) for x in args.seeds.split(",") if x.strip()]:
        trainable, summary = build_split(struct, seed)
        trainable.to_csv(
            DATA / f"modeling_structures_seed{seed}.csv", index=False,
            encoding="utf-8-sig",
        )
        payload = {
            k: trainable.loc[trainable["split"] == k, "model_index"]
            .sort_values().astype(int).tolist()
            for k in ("train", "val", "test")
        }
        payload["seed"] = seed
        payload["structure_ids"] = {
            k: trainable.loc[trainable["split"] == k, "structure_id"]
            .sort_values().tolist()
            for k in ("train", "val", "test")
        }
        dump_json(SPLITS / f"split_seed{seed}.json", payload)
        all_summary[str(seed)] = summary
        print(f"seed {seed}: " + ", ".join(
            f"{k}={v['n']} (pos {v['positive']}, "
            f"scaffolds {v['scaffolds']})"
            for k, v in summary["splits"].items()
        ))
    dump_json(SPLITS / "split_summary.json", all_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
