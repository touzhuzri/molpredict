"""Leakage and artifact audit for the APPf-like sub-experiment."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from appf_features import load_split  # noqa: E402
from common import DATA, OUT, dump_json  # noqa: E402

APPF = OUT / "appf_like"


def main() -> int:
    payload = {"splits": {}, "models": {}}
    for seed in (42, 43, 44):
        table = pd.read_csv(
            DATA / f"modeling_structures_seed{seed}.csv",
            low_memory=False,
        ).sort_values("model_index").reset_index(drop=True)
        split = load_split(seed, table)
        meta = {}
        for name, idx in split.items():
            sub = table.iloc[idx]
            meta[name] = {
                "n": int(len(sub)),
                "n_positive": int((sub["label"] == 1).sum()),
                "n_negative": int((sub["label"] == 0).sum()),
                "n_unknown": int(sub["label"].isna().sum()),
                "label_values": sorted(
                    str(x) for x in sub["label"].dropna().unique()
                ),
                "structure_ids": set(sub["structure_id"]),
                "scaffolds": set(sub["scaffold"]),
            }
        names = ["train", "val", "test"]
        overlaps = {}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                overlaps[f"{a}_{b}_structure"] = len(
                    meta[a]["structure_ids"] & meta[b]["structure_ids"]
                )
                overlaps[f"{a}_{b}_scaffold"] = len(
                    meta[a]["scaffolds"] & meta[b]["scaffolds"]
                )
        payload["splits"][str(seed)] = {
            name: {k: v for k, v in meta[name].items()
                   if k not in ("structure_ids", "scaffolds")}
            for name in names
        }
        payload["splits"][str(seed)]["overlaps"] = overlaps
        payload["splits"][str(seed)]["labels_binary"] = all(
            meta[name]["label_values"] in (["0"], ["1"], ["0", "1"])
            for name in names
        )
        payload["splits"][str(seed)]["no_unknown_or_conflict"] = all(
            meta[name]["n_unknown"] == 0 for name in names
        )

    for name in ["P24", "EXT", "EXT_OLD_ONLY"]:
        for seed in (42, 43, 44):
            d = APPF / "models" / name / f"split{seed}"
            if d.exists():
                payload["models"][f"{name}/split{seed}"] = len(
                    list(d.glob("bag*.cbm"))
                )
    dump_json(APPF / "metrics" / "isolation_audit.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
