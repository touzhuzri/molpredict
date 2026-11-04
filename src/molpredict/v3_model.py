"""V3 统一多模态 Transformer。

V3 不再把每个模态先池化成一个向量再融合。三个模态被组织成一条 token 序列：

    [CLS] + SMILES tokens + ConfSeq tokens + 24 descriptor tokens

所有 token 进入同一个多层 Transformer，使用同一套注意力矩阵和 padding mask。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from molpredict.v3_encoders import (
    MAX_CONFSEQ_LEN,
    MAX_SMILES_LEN,
    load_confseq_parts,
    load_smiles_encoder,
)

MODALITY_IDS = {"smiles": 0, "confseq": 1, "descriptors": 2}
DEFAULT_FEATURES = ("smiles", "confseq", "descriptors")


@dataclass
class V3Config:
    """可序列化的模型结构配置。"""

    modalities: tuple[str, ...] = DEFAULT_FEATURES
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    ff_mult: int = 2
    dropout: float = 0.2
    max_smiles_len: int = MAX_SMILES_LEN
    max_confseq_len: int = MAX_CONFSEQ_LEN
    n_descriptors: int = 24
    smiles_dim: int = 768
    confseq_dim: int = 256
    smiles_live: bool = False
    confseq_live: bool = False
    unfreeze_smiles_layers: int = 0
    unfreeze_confseq_layers: int = 0
    descriptor_value_hidden: int = 64
    pooling: str = "cls"  # "cls" 或 "cls_mean"

    def __post_init__(self):
        self.modalities = tuple(self.modalities)
        unknown = set(self.modalities) - set(MODALITY_IDS)
        if unknown:
            raise ValueError(f"未知模态: {unknown}")
        if not self.modalities:
            raise ValueError("至少需要一个模态")

    def to_dict(self) -> dict:
        data = asdict(self)
        data["modalities"] = list(self.modalities)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "V3Config":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


class TokenProjector(nn.Module):
    """把编码器的 token 表示投影到共同维度，并加入可学习位置。"""

    def __init__(self, in_dim: int, d_model: int, max_len: int,
                 dropout: float):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, d_model), nn.LayerNorm(d_model),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.pos = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) > self.pos.num_embeddings:
            raise ValueError(
                f"序列长度 {x.size(1)} 超过位置编码上限 "
                f"{self.pos.num_embeddings}"
            )
        pos = torch.arange(x.size(1), device=x.device)
        return self.proj(x) + self.pos(pos).unsqueeze(0)


class DescriptorTokenEncoder(nn.Module):
    """24 个描述符 → 24 个独立 token。

    每个 token = MLP(标准化的标量值) + 该特征的身份 embedding。
    24 项之间没有人为的“原子位置”对应关系，身份由特征顺序固定。
    """

    def __init__(self, n_descriptors: int, d_model: int, hidden: int,
                 dropout: float):
        super().__init__()
        self.value = nn.Sequential(
            nn.Linear(1, hidden), nn.GELU(),
            nn.Linear(hidden, d_model), nn.Dropout(dropout),
        )
        self.identity = nn.Embedding(n_descriptors, d_model)
        nn.init.normal_(self.identity.weight, std=0.02)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        # values: [B, n_desc]
        value_tokens = self.value(values.unsqueeze(-1))   # [B, n_desc, d]
        return value_tokens + self.identity.weight.unsqueeze(0)


class LiveSmilesEncoder(nn.Module):
    def __init__(self, unfreeze_last: int = 0):
        super().__init__()
        self.encoder = load_smiles_encoder(unfreeze_last)

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor):
        return self.encoder(
            input_ids=input_ids, attention_mask=mask.long()
        ).last_hidden_state


class LiveConfSeqEncoder(nn.Module):
    def __init__(self, unfreeze_last: int = 0):
        super().__init__()
        self.encoder, self.vocab_dict, self.pad_id, self.unk_id = (
            load_confseq_parts(unfreeze_last)
        )

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor):
        return self.encoder(input_ids=input_ids, attention_mask=mask.long())


class V3Net(nn.Module):
    """统一 token 级融合网络。"""

    def __init__(self, config: V3Config,
                 smiles_encoder: nn.Module | None = None,
                 confseq_encoder: nn.Module | None = None):
        super().__init__()
        self.config = config
        d = config.d_model
        self.modality_embed = nn.Embedding(3, d)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.cls_token, std=0.02)

        if "smiles" in config.modalities:
            self.smiles_projector = TokenProjector(
                config.smiles_dim, d, config.max_smiles_len, config.dropout
            )
            if config.smiles_live:
                self.smiles_encoder = smiles_encoder or LiveSmilesEncoder(
                    config.unfreeze_smiles_layers
                )
            else:
                self.smiles_encoder = None

        if "confseq" in config.modalities:
            self.confseq_projector = TokenProjector(
                config.confseq_dim, d, config.max_confseq_len, config.dropout
            )
            if config.confseq_live:
                self.confseq_encoder = confseq_encoder or LiveConfSeqEncoder(
                    config.unfreeze_confseq_layers
                )
            else:
                self.confseq_encoder = None

        if "descriptors" in config.modalities:
            self.descriptor_encoder = DescriptorTokenEncoder(
                config.n_descriptors, d, config.descriptor_value_hidden,
                config.dropout,
            )

        self.input_norm = nn.LayerNorm(d)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=config.n_heads,
            dim_feedforward=d * config.ff_mult,
            dropout=config.dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(
            layer, num_layers=config.n_layers,
            enable_nested_tensor=False,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d // 2), nn.GELU(),
            nn.Dropout(config.dropout), nn.Linear(d // 2, 1),
        )

    def embed(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """构造共同注意力输入：``[CLS] + 各模态 token`` 及有效位置 mask。"""
        parts, masks, modality_ids = [], [], []

        if "smiles" in self.config.modalities:
            if self.smiles_encoder is not None:
                ids, mask = batch["smiles_ids"]
                states = self.smiles_encoder(ids, mask)
            else:
                states = batch["smiles_states"]
                mask = batch["smiles_mask"]
            h = self.smiles_projector(states)
            h = h * mask.unsqueeze(-1).to(h.dtype)
            parts.append(h)
            masks.append(mask.bool())
            modality_ids.append(torch.full(
                mask.shape, MODALITY_IDS["smiles"], dtype=torch.long,
                device=mask.device,
            ))

        if "confseq" in self.config.modalities:
            if self.confseq_encoder is not None:
                ids, mask = batch["confseq_ids"]
                states = self.confseq_encoder(ids, mask)
            else:
                states = batch["confseq_states"]
                mask = batch["confseq_mask"]
            h = self.confseq_projector(states)
            h = h * mask.unsqueeze(-1).to(h.dtype)
            parts.append(h)
            masks.append(mask.bool())
            modality_ids.append(torch.full(
                mask.shape, MODALITY_IDS["confseq"], dtype=torch.long,
                device=mask.device,
            ))

        if "descriptors" in self.config.modalities:
            if "descriptors" not in batch:
                raise KeyError("缺少 descriptors")
            h = self.descriptor_encoder(batch["descriptors"])
            mask = torch.ones(h.shape[:2], dtype=torch.bool,
                              device=h.device)
            parts.append(h)
            masks.append(mask)
            modality_ids.append(torch.full(
                mask.shape, MODALITY_IDS["descriptors"], dtype=torch.long,
                device=h.device,
            ))

        if not parts:
            raise ValueError("没有可用模态")

        x = torch.cat(parts, dim=1)
        mask = torch.cat(masks, dim=1)
        mod_ids = torch.cat(modality_ids, dim=1)
        x = x + self.modality_embed(mod_ids)

        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        cls_mask = torch.ones(
            (x.size(0), 1), dtype=torch.bool, device=x.device
        )
        mask = torch.cat([cls_mask, mask], dim=1)
        x = self.input_norm(x)
        return x, mask

    def forward(self, batch: dict) -> torch.Tensor:
        x, mask = self.embed(batch)
        x = self.fusion(x, src_key_padding_mask=~mask)
        cls = x[:, 0]
        if self.config.pooling == "cls_mean":
            body = x[:, 1:]
            body_mask = mask[:, 1:].unsqueeze(-1).to(body.dtype)
            mean = (body * body_mask).sum(dim=1) / body_mask.sum(
                dim=1
            ).clamp_min(1.0)
            cls = cls + mean
        elif self.config.pooling != "cls":
            raise ValueError(f"未知 pooling: {self.config.pooling}")
        return self.classifier(cls).squeeze(-1)


def build_v3_model(config: V3Config, *, load_live_encoders: bool = True) -> V3Net:
    """构建模型；live 编码器在模型内部，以便随 checkpoint 保存/加载。"""
    if load_live_encoders:
        return V3Net(config)
    # 测试/预测缓存路径时可以不加载 44M 编码器
    if config.smiles_live or config.confseq_live:
        raise ValueError("live 编码器必须加载")
    return V3Net(config)


def count_trainable(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
