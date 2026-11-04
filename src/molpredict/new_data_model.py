"""新数据五组对照实验用的辅助学习模型与数据容器。

主模型仍是 V3Net；D 组在其上增加一个只读取 SMILES 表示的辅助头，
用于从结构预测 24 项 PaDEL。主任务输入不改变。
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from molpredict.v3_model import V3Config, V3Net


class AuxV3Net(nn.Module):
    """V3Net + structure→PaDEL 辅助头（辅助头不看 PaDEL token）。"""

    def __init__(self, config: V3Config, aux_dim: int = 24,
                 aux_hidden: int = 64, aux_dropout: float = 0.3):
        super().__init__()
        if "smiles" not in config.modalities:
            raise ValueError("辅助任务要求 SMILES 输入")
        self.backbone = V3Net(config)
        d = config.d_model
        self.aux_head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, aux_hidden),
            nn.GELU(),
            nn.Dropout(aux_dropout),
            nn.Linear(aux_hidden, aux_dim),
        )
        self.aux_dim = aux_dim

    def forward(self, batch: dict) -> torch.Tensor:
        return self.backbone(batch)

    def forward_aux(self, batch: dict) -> torch.Tensor:
        states = batch["smiles_states"]
        mask = batch["smiles_mask"].bool()
        h = self.backbone.smiles_projector(states)
        h = h * mask.unsqueeze(-1).to(h.dtype)
        pooled = h.sum(dim=1) / mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
        return self.aux_head(pooled)


class AuxDataset(Dataset):
    """只含 SMILES 表示和 PaDEL 目标；目标 NaN 由 mask 排除。"""

    def __init__(self, states: list[torch.Tensor],
                 targets, mask):
        self.states = states
        self.targets = targets
        self.mask = mask

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, i: int) -> dict:
        return {
            "smiles_state": self.states[i].float(),
            "aux_target": torch.tensor(self.targets[i], dtype=torch.float32),
            "aux_mask": torch.tensor(self.mask[i], dtype=torch.float32),
        }


def collate_aux(batch: list[dict]) -> dict:
    states = pad_sequence(
        [b["smiles_state"] for b in batch], batch_first=True, padding_value=0.0
    )
    mask = pad_sequence(
        [torch.ones(b["smiles_state"].size(0), dtype=torch.bool)
         for b in batch],
        batch_first=True, padding_value=False,
    )
    return {
        "smiles_states": states,
        "smiles_mask": mask,
        "aux_target": torch.stack([b["aux_target"] for b in batch]),
        "aux_mask": torch.stack([b["aux_mask"] for b in batch]),
    }
