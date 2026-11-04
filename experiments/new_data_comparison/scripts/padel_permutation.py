"""检查 C 组是否真正使用 24 项 PaDEL：测试期置换描述符。

行置换：把整个 24 维向量在测试分子之间打乱；
列置换：分别打乱每一列，保留其他列。
两种都只改变描述符，不改变 SMILES/模型/其他数据。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from molpredict.v3_data import (  # noqa: E402
    NumericStandardizer,
    V3Dataset,
    collate_v3,
    load_cache_arrays,
    load_rows,
)
from molpredict.v3_model import V3Config  # noqa: E402
from molpredict.v3_train import load_model_for_inference  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402


def _probs(model, dataset, batch_size=32):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_v3)
    probs, labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {k: (tuple(t.to("cuda") for t in v)
                         if isinstance(v, (tuple, list))
                         else v.to("cuda"))
                     for k, v in batch.items()}
            probs.extend(torch.sigmoid(model(batch)).cpu().numpy().tolist())
            labels.extend(batch["label"].cpu().numpy().tolist())
    return np.asarray(probs), np.asarray(labels).astype(int)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path,
                        default=ROOT / "experiments" / "new_data_comparison" /
                        "checkpoints" / "C_OLD_ONLY_seed42")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "experiments" / "new_data_comparison" /
                        "ablations" / "padel_permutation.json")
    parser.add_argument("--repeats", type=int, default=32)
    args = parser.parse_args()

    model, config = load_model_for_inference(args.model, device="cuda")
    split = json.loads((args.model / "split.json").read_text(encoding="utf-8"))
    std = NumericStandardizer.from_dict(json.loads(
        (args.model / "standardization.json").read_text(encoding="utf-8")
    ))
    _, labels, rows = load_rows()
    arrays = load_cache_arrays(
        ROOT / "experiments" / "v2" / "cache",
        ROOT / "experiments" / "v3" / "cache",
        [r["inchikey"] for r in rows],
    )
    test_idx = np.asarray(split["test"], dtype=int)
    descriptors = std.transform(arrays["descriptors"])

    def dataset_for(desc_matrix):
        ds = V3Dataset(
            test_idx, labels,
            smiles_states=arrays["smiles_states"],
            smiles_ids=arrays["smiles_ids"],
            confseq_states=arrays["confseq_states"],
            confseq_ids=arrays["confseq_ids"],
            descriptors=desc_matrix,
        )
        return ds

    base_p, base_y = _probs(model, dataset_for(descriptors))
    base_auc = float(roc_auc_score(base_y, base_p))
    rng = np.random.default_rng(42)
    row_aucs, col_aucs = [], []
    for _ in range(args.repeats):
        perm = rng.permutation(len(test_idx))
        row_desc = descriptors.copy()
        row_desc[test_idx] = descriptors[test_idx][perm]
        row_aucs.append(float(roc_auc_score(
            base_y, _probs(model, dataset_for(row_desc))[0]
        )))
        col_desc = descriptors.copy()
        for j in range(col_desc.shape[1]):
            values = col_desc[test_idx, j]
            col_desc[test_idx, j] = values[rng.permutation(len(values))]
        col_aucs.append(float(roc_auc_score(
            base_y, _probs(model, dataset_for(col_desc))[0]
        )))

    result = {
        "model": str(args.model),
        "baseline_auc": base_auc,
        "row_permutation": {
            "mean": float(np.mean(row_aucs)),
            "std": float(np.std(row_aucs)),
            "min": float(np.min(row_aucs)),
            "max": float(np.max(row_aucs)),
            "drop": base_auc - float(np.mean(row_aucs)),
        },
        "column_permutation": {
            "mean": float(np.mean(col_aucs)),
            "std": float(np.std(col_aucs)),
            "min": float(np.min(col_aucs)),
            "max": float(np.max(col_aucs)),
            "drop": base_auc - float(np.mean(col_aucs)),
        },
        "repeats": args.repeats,
        "note": ("只置换测试期 PaDEL；SMILES、模型、训练/验证/测试划分不变。"
                 "AUC 下降说明模型依赖描述符；不下降说明没有真正使用。"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
