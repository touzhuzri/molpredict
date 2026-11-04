"""Test-time PaDEL permutation for the merged B model.

Only test PaDEL values are permuted; SMILES, labels and the model are fixed.
A large AUC drop means the model uses the descriptors.  A near-zero drop means
it effectively ignores them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import DATA, FEATURES, METRICS, SPLITS, dump_json  # noqa: E402
from modeling import (  # noqa: E402
    MergedDataset, build_model, collate_merged, move_batch,
)
from molpredict.v3_data import NumericStandardizer  # noqa: E402
from molpredict.v3_model import V3Config  # noqa: E402


def probs(model, ds, config, device, batch_size=32):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_merged)
    out = []
    model.eval()
    with torch.no_grad():
        for raw in loader:
            batch = move_batch(raw, config, device)
            logits = model(batch)
            out.extend(torch.sigmoid(logits.float()).cpu().numpy().tolist())
    return np.asarray(out, dtype=float)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="B")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=32)
    parser.add_argument("--aux", action="store_true",
                        help="checkpoint contains the C auxiliary head")
    args = parser.parse_args()
    run_dir = (ROOT / "experiments" / "merged_fungicide" / "checkpoints" /
               args.experiment / f"split{args.split_seed}_seed{args.seed}")
    config_data = json.loads(
        (run_dir / "config.json").read_text(encoding="utf-8")
    )
    config = V3Config.from_dict(config_data["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config, aux=args.aux).to(device)
    model.load_state_dict(torch.load(run_dir / "model.pt",
                                     map_location=device,
                                     weights_only=True))
    model.eval()

    split = json.loads(
        (SPLITS / f"split_seed{args.split_seed}.json").read_text(
            encoding="utf-8")
    )
    table = pd.read_csv(
        DATA / f"modeling_structures_seed{args.split_seed}.csv",
        low_memory=False,
    ).sort_values("model_index").reset_index(drop=True)
    states = torch.load(FEATURES / "smiles_states.pt", weights_only=False)
    tokens = torch.load(FEATURES / "smiles_tokens.pt", weights_only=False)
    raw = np.load(FEATURES / "padel24.npy")
    standardizer = NumericStandardizer.from_dict(json.loads(
        (run_dir / "standardization.json").read_text(encoding="utf-8")
    ))
    desc = standardizer.transform(raw)
    labels = table["label"].to_numpy(dtype=np.float32)
    feature_idx = table["structure_index"].to_numpy(dtype=int)
    test_idx = np.asarray(split["test"], dtype=int)

    base_arrays = {
        "smiles_states": states["states"], "smiles_ids": tokens["input_ids"],
        "descriptors": desc, "raw_descriptors": raw,
    }
    base_ds = MergedDataset(test_idx, labels, feature_idx, base_arrays)
    base_p = probs(model, base_ds, config, device)
    y = labels[test_idx].astype(int)
    base_auc = float(roc_auc_score(y, base_p))

    rng = np.random.default_rng(args.seed)
    row_aucs, col_aucs = [], []
    for _ in range(args.repeats):
        row_desc = desc.copy()
        perm = rng.permutation(len(test_idx))
        row_desc[feature_idx[test_idx]] = desc[feature_idx[test_idx]][perm]
        arrays = dict(base_arrays)
        arrays["descriptors"] = row_desc
        ds = MergedDataset(test_idx, labels, feature_idx, arrays)
        row_aucs.append(float(roc_auc_score(y, probs(model, ds, config,
                                                      device))))

        col_desc = desc.copy()
        for j in range(col_desc.shape[1]):
            vals = col_desc[feature_idx[test_idx], j]
            col_desc[feature_idx[test_idx], j] = vals[
                rng.permutation(len(vals))
            ]
        arrays = dict(base_arrays)
        arrays["descriptors"] = col_desc
        ds = MergedDataset(test_idx, labels, feature_idx, arrays)
        col_aucs.append(float(roc_auc_score(y, probs(model, ds, config,
                                                      device))))

    result = {
        "experiment": args.experiment,
        "split_seed": args.split_seed,
        "seed": args.seed,
        "baseline_auc": base_auc,
        "row_permutation": {
            "mean": float(np.mean(row_aucs)), "std": float(np.std(row_aucs)),
            "min": float(np.min(row_aucs)), "max": float(np.max(row_aucs)),
            "drop": base_auc - float(np.mean(row_aucs)),
        },
        "column_permutation": {
            "mean": float(np.mean(col_aucs)), "std": float(np.std(col_aucs)),
            "min": float(np.min(col_aucs)), "max": float(np.max(col_aucs)),
            "drop": base_auc - float(np.mean(col_aucs)),
        },
        "repeats": args.repeats,
        "note": "Only test-time PaDEL is permuted; SMILES and labels are fixed.",
    }
    dump_json(METRICS / f"padel_permutation_{args.experiment}_"
                        f"split{args.split_seed}_seed{args.seed}.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
