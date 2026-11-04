"""V3 数据：复用 V2 的 ConfSeq/PaDEL 数据，生成 V3 的完整 SMILES token 缓存。

数据隔离规则：
- scaffold 划分固定为 V2 的 seed 42/43/44；
- 标准化和缺失值中位数只在训练集拟合；
- 测试集不参与早停或阈值选择。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from molpredict.modeling import load_dataset, scaffold_groups
from molpredict.padel import PADEL_NAMES
from molpredict.v3_encoders import (
    CONFSEQ_PAD_ID,
    MAX_CONFSEQ_LEN,
    MAX_SMILES_LEN,
    cache_confseq_states,
    cache_smiles_states,
    sha256,
)

ROOT = Path(__file__).resolve().parents[2]
TRAIN_CSV = ROOT / "data" / "training" / "confseq.csv"
V2_CACHE = ROOT / "experiments" / "v2" / "cache"
V3_CACHE = ROOT / "experiments" / "v3" / "cache"
V2_SPLITS = ROOT / "experiments" / "v2" / "splits"


def load_rows(csv_path: Path = TRAIN_CSV):
    mols, labels, rows = load_dataset(csv_path)
    return mols, labels, rows


def load_split(seed: int, source_dir: Path = V2_SPLITS) -> dict:
    path = source_dir / f"split_seed{seed}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    split = {k: np.asarray(data[k], dtype=int) for k in ("train", "val", "test")}
    all_idx = np.concatenate([split["train"], split["val"], split["test"]])
    if len(np.unique(all_idx)) != len(all_idx):
        raise ValueError(f"{path} 存在重叠索引")
    return split


def _verify_inchikeys(payload: dict, rows, name: str) -> None:
    keys = payload.get("inchikeys")
    expected = [r["inchikey"] for r in rows]
    if keys != expected:
        raise ValueError(f"{name} 的 InChIKey 顺序与训练数据不一致")


def load_cache_arrays(v2_cache: Path = V2_CACHE,
                      v3_cache: Path = V3_CACHE,
                      expected_inchikeys: list[str] | None = None) -> dict:
    """加载 V3 SMILES 缓存 + V2 ConfSeq / PaDEL 缓存，并做对齐校验。"""
    smiles_path = v3_cache / "smiles_states.pt"
    smiles_tokens_path = v3_cache / "smiles_tokens.pt"
    confseq_path = v2_cache / "confseq_states.pt"
    confseq_tokens_path = v2_cache / "confseq_tokens.pt"
    padel_path = v2_cache / "padel24.csv"
    padel_order_path = v2_cache / "padel24_order.json"

    for p in (smiles_path, smiles_tokens_path, confseq_path,
              confseq_tokens_path, padel_path, padel_order_path):
        if not p.exists():
            raise FileNotFoundError(
                f"缺少缓存 {p}；先运行 run_v3_experiments.py --prepare"
            )

    smiles = torch.load(smiles_path, weights_only=False)
    smiles_tokens = torch.load(smiles_tokens_path, weights_only=False)
    confseq = torch.load(confseq_path, weights_only=False)
    confseq_tokens = torch.load(confseq_tokens_path, weights_only=False)
    for payload, name in (
        (smiles, "smiles_states"), (smiles_tokens, "smiles_tokens"),
        (confseq, "confseq_states"), (confseq_tokens, "confseq_tokens"),
    ):
        if payload.get("inchikeys") is None:
            raise ValueError(f"{name} 缺少 inchikeys")
        if (expected_inchikeys is not None
                and payload["inchikeys"] != expected_inchikeys):
            raise ValueError(f"{name} 的 InChIKey 顺序与训练数据不一致")

    import pandas as pd
    df = pd.read_csv(padel_path)
    if len(df) != len(smiles["inchikeys"]):
        raise ValueError("PaDEL 缓存行数与 SMILES 缓存不一致")
    missing = [n for n in PADEL_NAMES if n not in df.columns]
    if missing:
        raise ValueError(f"PaDEL 缓存缺少列: {missing}")
    order = json.loads(padel_order_path.read_text(encoding="utf-8"))
    if order.get("inchikeys") != smiles["inchikeys"]:
        raise ValueError("PaDEL 顺序文件与 SMILES 缓存不一致")

    return {
        "smiles_states": smiles["states"],
        "smiles_ids": smiles_tokens["input_ids"],
        "confseq_states": confseq["states"],
        "confseq_ids": confseq_tokens["input_ids"],
        "descriptors": df[PADEL_NAMES].to_numpy(dtype=np.float32),
        "inchikeys": smiles["inchikeys"],
    }


def prepare_v3_cache(csv_path: Path = TRAIN_CSV,
                     v2_cache: Path = V2_CACHE,
                     v3_cache: Path = V3_CACHE) -> dict:
    """生成 V3 的 320-token SMILES 缓存，并校验/复用 V2 的 ConfSeq、PaDEL。"""
    _, _, rows = load_rows(csv_path)
    v3_cache.mkdir(parents=True, exist_ok=True)
    cache_smiles_states(rows, v3_cache, max_length=MAX_SMILES_LEN)
    if not (v2_cache / "confseq_states.pt").exists():
        cache_confseq_states(rows, v2_cache)
    # 校验所有缓存行序
    arrays = load_cache_arrays(
        v2_cache, v3_cache, expected_inchikeys=[r["inchikey"] for r in rows]
    )
    manifest = {
        "training_csv": str(csv_path),
        "training_csv_sha256": sha256(csv_path),
        "n_rows": len(rows),
        "smiles_cache": str(v3_cache / "smiles_states.pt"),
        "smiles_cache_sha256": sha256(v3_cache / "smiles_states.pt"),
        "smiles_max_length": MAX_SMILES_LEN,
        "confseq_cache": str(v2_cache / "confseq_states.pt"),
        "confseq_cache_sha256": sha256(v2_cache / "confseq_states.pt"),
        "confseq_max_length": MAX_CONFSEQ_LEN,
        "padel_cache": str(v2_cache / "padel24.csv"),
        "padel_cache_sha256": sha256(v2_cache / "padel24.csv"),
        "padel_names": PADEL_NAMES,
        "n_descriptor_columns": len(PADEL_NAMES),
        "descriptor_nan_counts": {
            name: int(np.isnan(arrays["descriptors"][:, i]).sum())
            for i, name in enumerate(PADEL_NAMES)
        },
    }
    (v3_cache / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


@dataclass
class NumericStandardizer:
    median: np.ndarray
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray, train_idx: np.ndarray) -> "NumericStandardizer":
        train = np.asarray(X, dtype=np.float64)[train_idx]
        median = np.nanmedian(train, axis=0)
        median[~np.isfinite(median)] = 0.0
        filled = np.where(np.isfinite(X), X, median)
        mean = filled[train_idx].mean(axis=0)
        std = filled[train_idx].std(axis=0)
        std[std < 1e-6] = 1.0
        return cls(median=median, mean=mean, std=std)

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        filled = np.where(np.isfinite(X), X, self.median)
        return ((filled - self.mean) / self.std).astype(np.float32)

    def to_dict(self) -> dict:
        return {
            "median": self.median.tolist(),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "fit_on": "training split only",
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NumericStandardizer":
        return cls(
            median=np.asarray(data["median"], dtype=np.float64),
            mean=np.asarray(data["mean"], dtype=np.float64),
            std=np.asarray(data["std"], dtype=np.float64),
        )


class V3Dataset(Dataset):
    def __init__(self, indices, labels, *, smiles_states, smiles_ids,
                 confseq_states, confseq_ids, descriptors):
        self.indices = np.asarray(indices, dtype=int)
        self.labels = labels
        self.smiles_states = smiles_states
        self.smiles_ids = smiles_ids
        self.confseq_states = confseq_states
        self.confseq_ids = confseq_ids
        self.descriptors = descriptors

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict:
        idx = int(self.indices[i])
        return {
            "label": torch.tensor(float(self.labels[idx]), dtype=torch.float32),
            "smiles_state": self.smiles_states[idx].float(),
            "smiles_id": self.smiles_ids[idx].long(),
            "confseq_state": self.confseq_states[idx].float(),
            "confseq_id": self.confseq_ids[idx].long(),
            "descriptors": torch.tensor(self.descriptors[idx], dtype=torch.float32),
        }


def _pad_tokens(items, pad_value):
    mask = pad_sequence(
        [torch.ones(x.size(0), dtype=torch.bool) for x in items],
        batch_first=True, padding_value=False,
    )
    padded = pad_sequence(items, batch_first=True, padding_value=pad_value)
    return padded, mask


def collate_v3(batch: list[dict]) -> dict:
    out = {"label": torch.stack([b["label"] for b in batch])}
    states, mask = _pad_tokens([b["smiles_state"] for b in batch], 0.0)
    out["smiles_states"] = states
    out["smiles_mask"] = mask
    out["smiles_ids"] = _pad_tokens([b["smiles_id"] for b in batch], 1)
    states, mask = _pad_tokens([b["confseq_state"] for b in batch], 0.0)
    out["confseq_states"] = states
    out["confseq_mask"] = mask
    out["confseq_ids"] = _pad_tokens(
        [b["confseq_id"] for b in batch], CONFSEQ_PAD_ID
    )
    out["descriptors"] = torch.stack([b["descriptors"] for b in batch])
    return out


def leakage_audit(mols, rows, split: dict, seed: int) -> dict:
    """检查数据内泄漏；预训练语料重叠只能标记为未确认。"""
    from rdkit import Chem

    canonical = []
    for row in rows:
        mol = Chem.MolFromSmiles(row["smiles"])
        canonical.append(Chem.MolToSmiles(mol, isomericSmiles=True)
                         if mol is not None else None)
    groups = scaffold_groups(mols)
    inchi = [r["inchikey"] for r in rows]
    canon_counts = {}
    for s in canonical:
        canon_counts[s] = canon_counts.get(s, 0) + 1
    inchi_counts = {}
    for k in inchi:
        inchi_counts[k] = inchi_counts.get(k, 0) + 1

    sets = {k: {int(i) for i in v} for k, v in split.items()}
    overlaps = {}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlaps[f"{a}_{b}_indices"] = len(sets[a] & sets[b])
        overlaps[f"{a}_{b}_inchikey"] = len(
            {inchi[i] for i in sets[a]} & {inchi[i] for i in sets[b]}
        )
        overlaps[f"{a}_{b}_scaffold"] = len(
            {groups[i] for i in sets[a]} & {groups[i] for i in sets[b]}
        )
    return {
        "seed": seed,
        "n_molecules": len(rows),
        "n_positive": int(sum(int(r["label"]) for r in rows)),
        "duplicate_inchikey": int(sum(v > 1 for v in inchi_counts.values())),
        "duplicate_canonical_smiles": int(
            sum(v > 1 for v in canon_counts.values())
        ),
        "split_overlaps": overlaps,
        "pretraining_overlap": {
            "status": "unknown",
            "reason": (
                "未获得 ChemBERTa/ZINC 与 ConfSeq 预训练语料的完整分子清单，"
                "无法逐分子审计；不能声称不存在重叠。"
            ),
        },
    }
