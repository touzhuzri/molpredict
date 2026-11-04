"""Mechanical checks for structure grouping and auxiliary leakage."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import DATA, METRICS, SPLITS, dump_json  # noqa: E402


def main() -> int:
    struct = pd.read_csv(DATA / "structures.csv", low_memory=False)
    report = {"splits": {}, "checks": {}}
    for seed in (42, 43, 44):
        table = pd.read_csv(
            DATA / f"modeling_structures_seed{seed}.csv", low_memory=False
        ).sort_values("model_index")
        split = json.loads(
            (SPLITS / f"split_seed{seed}.json").read_text(encoding="utf-8")
        )
        ids = {
            k: set(table.iloc[split[k]]["structure_id"]) for k in split
            if k in {"train", "val", "test"}
        }
        scafs = {
            k: set(table.iloc[split[k]]["scaffold"]) for k in split
            if k in {"train", "val", "test"}
        }
        report["splits"][str(seed)] = {
            "n": {k: len(v) for k, v in ids.items()},
            "structure_overlap": {
                "train_val": len(ids["train"] & ids["val"]),
                "train_test": len(ids["train"] & ids["test"]),
                "val_test": len(ids["val"] & ids["test"]),
            },
            "scaffold_overlap": {
                "train_val": len(scafs["train"] & scafs["val"]),
                "train_test": len(scafs["train"] & scafs["test"]),
                "val_test": len(scafs["val"] & scafs["test"]),
            },
            "aux_scaffold_overlap_val": len(
                (set(struct["scaffold"]) - scafs["val"] - scafs["test"])
                & scafs["val"]
            ),
            "aux_scaffold_overlap_test": len(
                (set(struct["scaffold"]) - scafs["val"] - scafs["test"])
                & scafs["test"]
            ),
        }
        all_idx = split["train"] + split["val"] + split["test"]
        report["checks"][f"seed{seed}"] = {
            "all_supervised_indices_unique": len(all_idx) == len(set(all_idx)),
            "all_labels_binary": bool(
                table["label"].isin([0, 1]).all()
            ),
            "no_unknown_or_conflict": bool(
                table["label_status"].isin(["POSITIVE", "NEGATIVE"]).all()
            ),
            "structure_id_unique": bool(
                table["structure_id"].is_unique
            ),
        }
    dump_json(METRICS / "isolation_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
