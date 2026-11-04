"""Dataset and model wrappers for the merged fungicide experiments."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from molpredict.v3_model import V3Config, V3Net  # noqa: E402


class MergedDataset(Dataset):
    def __init__(self, indices, labels, feature_indices, arrays):
        self.indices = np.asarray(indices, dtype=int)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.feature_indices = np.asarray(feature_indices, dtype=int)
        self.states = arrays["smiles_states"]
        self.ids = arrays["smiles_ids"]
        self.descriptors = arrays["descriptors"]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict:
        row = int(self.indices[i])
        feat = int(self.feature_indices[row])
        return {
            "label": torch.tensor(self.labels[row], dtype=torch.float32),
            "smiles_state": self.states[feat].float(),
            "smiles_id": self.ids[feat].long(),
            "descriptors": torch.tensor(
                self.descriptors[feat], dtype=torch.float32
            ),
        }


class AuxDataset(Dataset):
    """All parseable structures for SMILES -> PaDEL auxiliary learning."""

    def __init__(self, feature_indices, arrays):
        self.feature_indices = np.asarray(feature_indices, dtype=int)
        self.states = arrays["smiles_states"]
        self.ids = arrays["smiles_ids"]
        self.descriptors = arrays["descriptors"]
        self.raw_descriptors = arrays["raw_descriptors"]

    def __len__(self) -> int:
        return len(self.feature_indices)

    def __getitem__(self, i: int) -> dict:
        feat = int(self.feature_indices[i])
        raw = self.raw_descriptors[feat]
        return {
            "smiles_state": self.states[feat].float(),
            "smiles_id": self.ids[feat].long(),
            "aux_target": torch.tensor(
                self.descriptors[feat], dtype=torch.float32
            ),
            "aux_mask": torch.tensor(
                np.isfinite(raw).astype(np.float32), dtype=torch.float32
            ),
        }


def _pad_tokens(items, pad_value):
    mask = pad_sequence(
        [torch.ones(x.size(0), dtype=torch.bool) for x in items],
        batch_first=True, padding_value=False,
    )
    padded = pad_sequence(items, batch_first=True, padding_value=pad_value)
    return padded, mask


def collate_merged(batch: list[dict]) -> dict:
    out = {"label": torch.stack([b["label"] for b in batch])}
    states, mask = _pad_tokens([b["smiles_state"] for b in batch], 0.0)
    out["smiles_states"] = states
    out["smiles_mask"] = mask
    out["smiles_ids"] = _pad_tokens(
        [b["smiles_id"] for b in batch], 1
    )[0]
    out["descriptors"] = torch.stack([b["descriptors"] for b in batch])
    return out


def collate_aux(batch: list[dict]) -> dict:
    states, mask = _pad_tokens([b["smiles_state"] for b in batch], 0.0)
    return {
        "smiles_states": states,
        "smiles_mask": mask,
        "aux_target": torch.stack([b["aux_target"] for b in batch]),
        "aux_mask": torch.stack([b["aux_mask"] for b in batch]),
    }


class AuxNet(nn.Module):
    """V3 backbone plus a SMILES -> 24 PaDEL auxiliary head."""

    def __init__(self, config: V3Config, hidden: int = 64):
        super().__init__()
        self.backbone = V3Net(config)
        d = config.d_model
        self.aux_head = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, hidden), nn.GELU(),
            nn.Dropout(config.dropout), nn.Linear(hidden, config.n_descriptors),
        )

    def forward(self, batch: dict) -> torch.Tensor:
        return self.backbone(batch)

    def forward_aux(self, batch: dict) -> torch.Tensor:
        states = batch["smiles_states"]
        mask = batch["smiles_mask"].bool()
        h = self.backbone.smiles_projector(states)
        h = h * mask.unsqueeze(-1).to(h.dtype)
        pooled = h.sum(dim=1) / mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
        return self.aux_head(pooled)


def build_model(config: V3Config, aux: bool = False) -> nn.Module:
    return AuxNet(config) if aux else V3Net(config)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total), int(trainable)


def move_batch(raw: dict, config: V3Config, device: torch.device) -> dict:
    keys = ["label"]
    if "smiles" in config.modalities:
        keys += ["smiles_states", "smiles_mask"]
    if "descriptors" in config.modalities:
        keys += ["descriptors"]
    out = {}
    for k in keys:
        v = raw[k]
        if isinstance(v, (tuple, list)):
            out[k] = tuple(x.to(device, non_blocking=True) for x in v)
        else:
            out[k] = v.to(device, non_blocking=True)
    return out


def move_aux(raw: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in raw.items()}


def masked_mse(pred: torch.Tensor, target: torch.Tensor,
               mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float()
    sq = (pred.float() - target.float()) ** 2
    return (sq * mask).sum() / mask.sum().clamp_min(1.0)
