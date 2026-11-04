"""V3.1 训练：GPU/AMP、更深融合网络、逐轮学习诊断。

与 V3 的关系：
- 模型结构仍是同一个 ``[CLS] + SMILES + ConfSeq + 24 descriptors`` 共同注意力；
- V3.1 只改变训练配方（设备、混合精度、调度、轮数）并允许更深的配置；
- 所有 V3 原始产物保持不变，V3.1 写入独立目录。
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from molpredict.padel import PADEL_NAMES
from molpredict.v2 import best_threshold, binary_metrics
from molpredict.v3_data import (
    NumericStandardizer,
    V3Dataset,
    collate_v3,
    load_cache_arrays,
    load_rows,
)
from molpredict.v3_encoders import (
    pretrained_manifest,
    save_confseq_vocab,
    save_tokenizer,
    sha256,
)
from molpredict.v3_model import V3Config, V3Net, count_trainable

ROOT = Path(__file__).resolve().parents[2]
V2_CACHE = ROOT / "experiments" / "v2" / "cache"
V3_CACHE = ROOT / "experiments" / "v3" / "cache"


# ---------------------------------------------------------------------------
# 设备
# ---------------------------------------------------------------------------


def resolve_device(requested: str = "auto", require_cuda: bool = False):
    if requested == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        name = requested
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "请求了 CUDA，但 torch.cuda.is_available() 为 False；"
            f"当前 torch={torch.__version__}, "
            f"torch.version.cuda={torch.version.cuda}"
        )
    if require_cuda and device.type != "cuda":
        raise RuntimeError(
            "正式 GPU 实验要求 CUDA 可用；拒绝静默退回 CPU"
        )
    return device


def gpu_info(device: torch.device | None = None) -> dict:
    if not torch.cuda.is_available():
        return {"available": False, "torch": torch.__version__,
                "torch_cuda": torch.version.cuda}
    idx = 0 if device is None or device.index is None else device.index
    prop = torch.cuda.get_device_properties(idx)
    return {
        "available": True,
        "name": prop.name,
        "total_memory_gb": round(prop.total_memory / 1024 ** 3, 2),
        "capability": list(torch.cuda.get_device_capability(idx)),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "device_index": idx,
    }


def nvidia_smi_status() -> dict:
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            timeout=5, text=True,
        ).strip().splitlines()[0]
        util, used, total = [int(x.strip()) for x in out.split(",")[:3]]
        return {"utilization_pct": util, "memory_used_mb": used,
                "memory_total_mb": total}
    except Exception as exc:  # 驱动/权限问题不应中断训练
        return {"error": str(exc)[:200]}


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------


def _loader(split, labels, arrays, standardizer, batch_size, device, seed):
    descriptors = standardizer.transform(arrays["descriptors"])
    ds = {
        k: V3Dataset(
            split[k], labels,
            smiles_states=arrays["smiles_states"],
            smiles_ids=arrays["smiles_ids"],
            confseq_states=arrays["confseq_states"],
            confseq_ids=arrays["confseq_ids"],
            descriptors=descriptors,
        ) for k in ("train", "val", "test")
    }
    gen = torch.Generator()
    gen.manual_seed(seed)
    pin = device.type == "cuda"
    return {
        "train": DataLoader(ds["train"], batch_size=batch_size, shuffle=True,
                            collate_fn=collate_v3, generator=gen,
                            pin_memory=pin),
        "val": DataLoader(ds["val"], batch_size=batch_size, shuffle=False,
                          collate_fn=collate_v3, pin_memory=pin),
        "test": DataLoader(ds["test"], batch_size=batch_size, shuffle=False,
                           collate_fn=collate_v3, pin_memory=pin),
    }


def batch_keys(config: V3Config) -> tuple[str, ...]:
    keys = ["label"]
    if "smiles" in config.modalities:
        keys.append("smiles_ids" if config.smiles_live else "smiles_states")
        if not config.smiles_live:
            keys.append("smiles_mask")
    if "confseq" in config.modalities:
        keys.append("confseq_ids" if config.confseq_live
                    else "confseq_states")
        if not config.confseq_live:
            keys.append("confseq_mask")
    if "descriptors" in config.modalities:
        keys.append("descriptors")
    return tuple(keys)


def move_batch(raw: dict, config: V3Config, device: torch.device) -> dict:
    """只传当前配置需要的张量，减少 CPU→GPU 流量。"""
    out = {}
    for key in batch_keys(config):
        value = raw[key]
        if isinstance(value, (tuple, list)):
            out[key] = tuple(
                t.to(device, non_blocking=True) for t in value
            )
        else:
            out[key] = value.to(device, non_blocking=True)
    return out


# ---------------------------------------------------------------------------
# 优化器与调度
# ---------------------------------------------------------------------------


def move_aux_batch(raw: dict, device: torch.device) -> dict:
    return {
        "smiles_states": raw["smiles_states"].to(device, non_blocking=True),
        "smiles_mask": raw["smiles_mask"].to(device, non_blocking=True),
        "aux_target": raw["aux_target"].to(device, non_blocking=True),
        "aux_mask": raw["aux_mask"].to(device, non_blocking=True),
    }


def masked_mse(pred: torch.Tensor, target: torch.Tensor,
               mask: torch.Tensor) -> torch.Tensor:
    """只对非 NaN 的 PaDEL 目标计算 MSE。"""
    mask = mask.float()
    sq = (pred.float() - target.float()) ** 2
    return (sq * mask).sum() / mask.sum().clamp_min(1.0)


def _cycle(loader):
    while True:
        for batch in loader:
            yield batch


def build_optimizer_groups(model: V3Net, lr: float, encoder_lr: float,
                           weight_decay: float):
    encoder_params = []
    for name in ("smiles_encoder", "confseq_encoder"):
        module = getattr(model, name, None)
        if module is None:
            continue
        encoder_params.extend(
            [p for p in module.parameters() if p.requires_grad]
        )
    encoder_ids = {id(p) for p in encoder_params}
    head_params = [
        p for p in model.parameters()
        if p.requires_grad and id(p) not in encoder_ids
    ]
    groups = [{"params": head_params, "lr": lr, "name": "fusion"}]
    if encoder_params:
        groups.append({"params": encoder_params, "lr": encoder_lr,
                       "name": "encoder"})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def warmup_cosine(optimizer, total_steps: int, warmup_ratio: float,
                  min_lr_ratio: float):
    warmup = max(1, int(total_steps * warmup_ratio))
    min_lr_ratio = max(0.0, min(1.0, min_lr_ratio))

    def fn(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(total_steps - warmup, 1)
        progress = min(max(progress, 0.0), 1.0)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (
            1 + math.cos(math.pi * progress)
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


# ---------------------------------------------------------------------------
# 诊断
# ---------------------------------------------------------------------------


def _group_name(param_name: str) -> str:
    if param_name.startswith("smiles_encoder."):
        return "smiles_encoder"
    if param_name.startswith("confseq_encoder."):
        return "confseq_encoder"
    if param_name.startswith("smiles_projector."):
        return "smiles_projector"
    if param_name.startswith("confseq_projector."):
        return "confseq_projector"
    if param_name.startswith("descriptor_encoder."):
        return "descriptor_encoder"
    if param_name.startswith("fusion.layers."):
        parts = param_name.split(".")
        return f"fusion_layer_{parts[2]}"
    if param_name.startswith("classifier."):
        return "classifier"
    if param_name.startswith("backbone."):
        rest = param_name.split(".", 1)[1]
        return _group_name(rest)
    if param_name.startswith("aux_head."):
        return "aux_head"
    return "embeddings"


def gradient_group_norms(model: nn.Module) -> dict:
    """按模块组统计梯度 L2 范数；只在梯度存在时调用。"""
    sq = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        key = _group_name(name)
        sq[key] = sq.get(key, 0.0) + float(
            p.grad.detach().float().pow(2).sum().item()
        )
    norms = {k: math.sqrt(v) for k, v in sq.items()}
    norms["total"] = math.sqrt(sum(v * v for k, v in norms.items()
                                  if k != "total"))
    return norms


def parameter_breakdown(model: nn.Module) -> dict:
    """按模块统计总参数 / 可训练参数。"""
    rows = {}
    for name, p in model.named_parameters():
        key = _group_name(name)
        row = rows.setdefault(key, {"total": 0, "trainable": 0})
        row["total"] += p.numel()
        row["trainable"] += p.numel() if p.requires_grad else 0
    total = sum(r["total"] for r in rows.values())
    trainable = sum(r["trainable"] for r in rows.values())
    return {
        "modules": dict(sorted(rows.items())),
        "total": int(total),
        "trainable": int(trainable),
        "frozen": int(total - trainable),
    }


def _modality_positions(batch: dict, config: V3Config):
    """每个样本在共同序列中的模态位置；返回 {样本: {模态: index array}}。"""
    bsz = batch["label"].size(0)
    layout = []
    offset = 1  # CLS
    if "smiles" in config.modalities:
        if config.smiles_live:
            ids, mask = batch["smiles_ids"]
        else:
            mask = batch["smiles_mask"]
        width = mask.size(1)
        layout.append(("smiles", offset, width, mask))
        offset += width
    if "confseq" in config.modalities:
        if config.confseq_live:
            ids, mask = batch["confseq_ids"]
        else:
            mask = batch["confseq_mask"]
        width = mask.size(1)
        layout.append(("confseq", offset, width, mask))
        offset += width
    if "descriptors" in config.modalities:
        width = config.n_descriptors
        layout.append(("descriptors", offset, width, None))
        offset += width

    out = []
    for b in range(bsz):
        sample = {}
        for name, start, width, mask in layout:
            if mask is None:
                idx = np.arange(start, start + width)
            else:
                valid = mask[b].bool().cpu().numpy()
                idx = start + np.nonzero(valid)[0]
            sample[name] = idx
        out.append(sample)
    return out, offset


def _attention_weights(x_in: torch.Tensor, mha: nn.MultiheadAttention,
                       key_padding_mask: torch.Tensor) -> torch.Tensor:
    """按 nn.MultiheadAttention 的定义精确重算注意力权重 [B,H,L,L]。"""
    bsz, length, dim = x_in.shape
    heads = mha.num_heads
    head_dim = dim // heads
    weight = mha.in_proj_weight
    bias = mha.in_proj_bias
    qkv = F.linear(x_in, weight, bias)
    q, k, _ = qkv.chunk(3, dim=-1)
    q = q.view(bsz, length, heads, head_dim).transpose(1, 2)
    k = k.view(bsz, length, heads, head_dim).transpose(1, 2)
    scores = q @ k.transpose(-2, -1) / math.sqrt(head_dim)
    if key_padding_mask is not None:
        scores = scores.masked_fill(
            key_padding_mask[:, None, None, :], float("-inf")
        )
    return torch.softmax(scores, dim=-1)


class AttentionAuditor:
    """在融合层 self-attention 上累计“源模态→目标模态”的注意力份额。

    只说明计算时的信息参考，不等于化学作用位点。
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        # acc[layer][src][tgt] = 每头累计和; count[layer]
        self.acc = {}
        self.count = {}
        self.n_samples = 0
        self.handles = []

    def attach(self, model: V3Net) -> None:
        for layer_idx, layer in enumerate(model.fusion.layers):
            handle = layer.self_attn.register_forward_pre_hook(
                self._make_hook(layer_idx), with_kwargs=True
            )
            self.handles.append(handle)

    def detach(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def set_context(self, positions, mask) -> None:
        self.positions = positions
        self.mask = mask
        self.limit = len(positions)

    def _make_hook(self, layer_idx: int):
        def hook(module, args, kwargs):
            x_in = args[0]
            kpm = kwargs.get("key_padding_mask")
            if kpm is None:
                kpm = torch.zeros(
                    x_in.size(0), x_in.size(1), dtype=torch.bool,
                    device=x_in.device,
                )
            elif kpm.dtype != torch.bool:
                # TransformerEncoder 可能把 padding mask 转成浮点表示
                kpm = kpm != 0
            with torch.no_grad():
                weights = _attention_weights(x_in, module, kpm)
                self._accumulate(layer_idx, weights)
        return hook

    def _accumulate(self, layer_idx: int, weights: torch.Tensor) -> None:
        # weights: [B, H, L, L]；每层只累计 limit 个样本
        bsz, heads = weights.shape[0], weights.shape[1]
        take = min(bsz, getattr(self, "limit", bsz))
        layer = self.acc.setdefault(layer_idx, {})
        for b in range(take):
            groups = self.positions[b]
            names = list(groups)
            for src in names:
                src_idx = torch.as_tensor(
                    groups[src], dtype=torch.long, device=weights.device
                )
                if src_idx.numel() == 0:
                    continue
                dst = layer.setdefault(src, {})
                for tgt in names:
                    tgt_idx = torch.as_tensor(
                        groups[tgt], dtype=torch.long, device=weights.device
                    )
                    if tgt_idx.numel() == 0:
                        continue
                    sub = weights[b][:, src_idx][:, :, tgt_idx]
                    # 每个源 token 到目标组的平均注意力 [H]
                    mass = sub.sum(dim=(1, 2)) / src_idx.numel()
                    dst[tgt] = dst.get(tgt, 0.0) + mass.cpu()
        self.count[layer_idx] = self.count.get(layer_idx, 0) + take
        self.n_samples = max(self.n_samples, self.count[layer_idx])

    def result(self, heads_by_layer: dict[int, int]) -> dict:
        layers = []
        for layer_idx in sorted(self.acc):
            heads = heads_by_layer.get(layer_idx, 1)
            count = max(self.count.get(layer_idx, 1), 1)
            block = {"layer": layer_idx, "heads": {}, "mean": {}}
            for src, dst in self.acc[layer_idx].items():
                per_head = {
                    k: (v / count).tolist()
                    for k, v in dst.items()
                }
                mean = {
                    k: sum(v) / len(v) for k, v in per_head.items()
                }
                total = sum(mean.values())
                block["mean"][src] = {
                    k: {"mass_per_token": v / heads,
                        "share": (v / heads) / total if total else None}
                    for k, v in mean.items()
                }
                for h in range(heads):
                    block["heads"].setdefault(str(h), {})[src] = {}
                    values = {k: v[h] for k, v in per_head.items()}
                    denom = sum(values.values())
                    for k, v in values.items():
                        block["heads"][str(h)][src][k] = {
                            "mass_per_token": v,
                            "share": v / denom if denom else None,
                        }
            layers.append(block)
        return {
            "n_samples": self.n_samples,
            "note": (
                "注意力份额只描述共同注意力计算时的信息参考，"
                "不能证明模型定位到真实化学作用位点。"
            ),
            "layers": layers,
        }


def attention_audit(model: V3Net, loader, config: V3Config,
                    device: torch.device, max_samples: int = 64) -> dict:
    model.eval()
    auditor = AttentionAuditor()
    auditor.attach(model)
    heads_by_layer = {i: layer.self_attn.num_heads
                      for i, layer in enumerate(model.fusion.layers)}
    seen = 0
    try:
        with torch.no_grad():
            for raw in loader:
                batch = move_batch(raw, config, device)
                x, mask = model.embed(batch)
                positions, _ = _modality_positions(batch, config)
                take = min(batch["label"].size(0), max_samples - seen)
                positions = positions[:take]
                auditor.set_context(positions, mask)
                # 触发融合层；hook 在每层 self-attention 前累计
                model.fusion(x, src_key_padding_mask=~mask)
                seen += take
                if seen >= max_samples:
                    break
    finally:
        auditor.detach()
    return auditor.result(heads_by_layer)


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------


def evaluate(model: V3Net, loader, criterion, config: V3Config,
             device: torch.device) -> dict:
    model.eval()
    probs, labels = [], []
    loss_sum = 0.0
    with torch.no_grad():
        for raw in loader:
            batch = move_batch(raw, config, device)
            logits = model(batch)
            loss = criterion(logits, batch["label"])
            loss_sum += float(loss.item()) * batch["label"].size(0)
            probs.extend(torch.sigmoid(logits).float().cpu().numpy().tolist())
            labels.extend(batch["label"].cpu().numpy().tolist())
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels).astype(int)
    return {
        "loss": loss_sum / max(len(labels), 1),
        "probs": probs,
        "labels": labels,
        "roc_auc": (_safe_auc(labels, probs)),
        "pr_auc": (_safe_pr(labels, probs)),
    }


def _safe_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return 0.0
    return float(roc_auc_score(labels, probs))


def _safe_pr(labels: np.ndarray, probs: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return 0.0
    return float(average_precision_score(labels, probs))


def _to_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False),
                    encoding="utf-8")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def code_manifest() -> dict:
    files = [
        ROOT / "src" / "molpredict" / "v3_model.py",
        ROOT / "src" / "molpredict" / "v3_data.py",
        ROOT / "src" / "molpredict" / "v3_encoders.py",
        ROOT / "src" / "molpredict" / "v3_1_train.py",
        ROOT / "scripts" / "run_v3_1_experiments.py",
    ]
    return {
        str(p.relative_to(ROOT)): sha256(p)
        for p in files if p.exists()
    }


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------


def train_v3_1(config: V3Config, split: dict, out_dir: Path, seed: int,
               *, epochs: int = 100, patience: int = 18,
               batch_size: int = 16, grad_accum: int = 2,
               lr: float = 3e-4, encoder_lr: float = 2e-5,
               weight_decay: float = 1e-4, warmup_ratio: float = 0.1,
               min_lr_ratio: float = 0.1, amp: str = "bf16",
               device_name: str = "auto", require_cuda: bool = True,
               attention_samples: int = 64,
               cache_dir: Path = V3_CACHE,
               model: nn.Module | None = None,
               aux_loader=None, aux_weight: float = 0.0,
               aux_source: str | None = None) -> dict:
    """训练一个 V3.1 run；测试集只在最后评估一次。"""
    _seed_everything(seed)
    n_threads = int(os.environ.get("MOLPREDICT_THREADS", "0"))
    if n_threads > 0:
        torch.set_num_threads(n_threads)
    device = resolve_device(device_name, require_cuda=require_cuda)
    amp = amp.lower()
    if amp not in {"bf16", "fp16", "off"}:
        raise ValueError(f"未知 amp: {amp}")
    use_amp = amp != "off" and device.type == "cuda"
    amp_dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    if amp == "bf16" and device.type == "cuda" \
            and not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU 不支持 BF16；请使用 --amp fp16")
    scaler = torch.amp.GradScaler(
        "cuda", enabled=use_amp and amp_dtype == torch.float16
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    mols, labels, rows = load_rows()
    expected = [r["inchikey"] for r in rows]
    arrays = load_cache_arrays(V2_CACHE, cache_dir, expected)
    split = {k: np.asarray(v, dtype=int) for k, v in split.items()}

    standardizer = NumericStandardizer.fit(
        arrays["descriptors"], split["train"]
    )
    loaders = _loader(split, labels, arrays, standardizer,
                      batch_size, device, seed)

    if model is None:
        model = V3Net(config)
    model = model.to(device)
    if aux_loader is not None and not hasattr(model, "forward_aux"):
        raise ValueError("aux_loader 需要带 forward_aux 的模型")
    breakdown = parameter_breakdown(model)
    total_params, trainable_params = breakdown["total"], breakdown["trainable"]
    pos = float(labels[split["train"]].sum())
    neg = float(len(split["train"]) - pos)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([neg / max(pos, 1.0)],
                                dtype=torch.float32, device=device)
    )
    optimizer = build_optimizer_groups(model, lr, encoder_lr, weight_decay)
    steps_per_epoch = math.ceil(len(loaders["train"]) / max(grad_accum, 1))
    total_steps = max(steps_per_epoch * epochs, 1)
    scheduler = warmup_cosine(optimizer, total_steps, warmup_ratio,
                              min_lr_ratio)

    history = []
    best_state, best_score, best_epoch, best_val = None, -1.0, -1, None
    bad = 0
    aux_iter = _cycle(aux_loader) if aux_loader is not None else None
    started = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        total_loss, seen = 0.0, 0
        aux_total, aux_seen = 0.0, 0
        n_batches = len(loaders["train"])
        last_grads = None
        optimizer.zero_grad(set_to_none=True)
        for step, raw in enumerate(loaders["train"], start=1):
            batch = move_batch(raw, config, device)
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=use_amp
            ):
                logits = model(batch)
            # 损失在 FP32 计算，避免 BF16/FP16 下的数值不稳定
            raw_loss = criterion(logits.float(), batch["label"])
            if aux_iter is not None:
                aux_batch = move_aux_batch(next(aux_iter), device)
                aux_pred = model.forward_aux(aux_batch)
                aux_loss = masked_mse(
                    aux_pred, aux_batch["aux_target"], aux_batch["aux_mask"]
                )
                loss = (raw_loss + aux_weight * aux_loss) / grad_accum
                aux_total += float(aux_loss.item()) * batch["label"].size(0)
                aux_seen += batch["label"].size(0)
            else:
                aux_loss = None
                loss = raw_loss / grad_accum
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            total_loss += float(raw_loss.item()) * batch["label"].size(0)
            seen += batch["label"].size(0)

            is_last = step == n_batches
            if step % grad_accum == 0 or is_last:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                grad_total = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 5.0
                )
                if is_last:
                    last_grads = gradient_group_norms(model)
                    last_grads["total_clipped"] = float(grad_total)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        train_loss = total_loss / max(seen, 1)
        train_eval = evaluate(model, loaders["train"], criterion, config,
                              device)
        val_eval = evaluate(model, loaders["val"], criterion, config, device)
        epoch_sec = time.time() - epoch_start
        lrs = {g.get("name", str(i)): g["lr"]
               for i, g in enumerate(optimizer.param_groups)}
        gpu = gpu_info(device)
        mem = {}
        if device.type == "cuda":
            mem = {
                "peak_allocated_mb": round(
                    torch.cuda.max_memory_allocated(device) / 1024 ** 2, 1
                ),
                "peak_reserved_mb": round(
                    torch.cuda.max_memory_reserved(device) / 1024 ** 2, 1
                ),
            }
        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_loss_eval": train_eval["loss"],
            "train_roc_auc": train_eval["roc_auc"],
            "train_pr_auc": train_eval["pr_auc"],
            "val_loss": val_eval["loss"],
            "val_roc_auc": val_eval["roc_auc"],
            "val_pr_auc": val_eval["pr_auc"],
            "lr": lrs,
            "aux_loss": (aux_total / max(aux_seen, 1)
                         if aux_iter is not None else None),
            "grad_norms": last_grads,
            "epoch_sec": round(epoch_sec, 2),
            "gpu": {**mem, **nvidia_smi_status()},
            "elapsed_sec": round(time.time() - started, 1),
        }
        history.append(entry)
        _to_json(out_dir / "history.json", history)
        print(
            f"[V3.1/{'+'.join(config.modalities)}/{config.d_model}d/"
            f"{config.n_layers}L/seed{seed}] epoch {epoch} "
            f"train_loss={train_loss:.4f} train_auc={train_eval['roc_auc']:.4f} "
            f"val_loss={val_eval['loss']:.4f} val_auc={val_eval['roc_auc']:.4f} "
            f"val_pr={val_eval['pr_auc']:.4f} lr={lrs.get('fusion', lr):.2e} "
            + (f"aux={aux_total / max(aux_seen, 1):.4f} "
               if aux_iter is not None else "")
            + f"{epoch_sec:.1f}s", flush=True,
        )

        score = val_eval["roc_auc"]
        if score > best_score + 1e-6:
            best_score, best_epoch, bad = score, epoch, 0
            best_val = val_eval
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
        else:
            bad += 1
            if bad >= patience:
                print(f"[V3.1/seed{seed}] early stop at epoch {epoch} "
                      f"(best {best_epoch}, val_auc={best_score:.4f})",
                      flush=True)
                break

    if best_state is None:
        raise RuntimeError("训练没有产生可用 checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    val_eval = evaluate(model, loaders["val"], criterion, config, device)
    threshold = best_threshold(val_eval["labels"], val_eval["probs"])
    val_metrics = binary_metrics(val_eval["labels"], val_eval["probs"],
                                 threshold)
    test_eval = evaluate(model, loaders["test"], criterion, config, device)
    test_metrics = binary_metrics(test_eval["labels"], test_eval["probs"],
                                  threshold)
    test_metrics["loss"] = test_eval["loss"]
    train_final = evaluate(model, loaders["train"], criterion, config, device)

    torch.save(best_state, out_dir / "model.pt")
    _to_json(out_dir / "standardization.json", standardizer.to_dict())
    _to_json(out_dir / "feature_order.json",
             {"padel24": PADEL_NAMES, "n": len(PADEL_NAMES)})
    if "smiles" in config.modalities:
        save_tokenizer(out_dir / "tokenizer")
    if "confseq" in config.modalities:
        save_confseq_vocab(out_dir)
    _to_json(out_dir / "split.json",
             {k: np.asarray(v).tolist() for k, v in split.items()})
    _to_json(out_dir / "pretrained.json", pretrained_manifest())

    attn = None
    if attention_samples > 0 and hasattr(model, "fusion"):
        attn = attention_audit(model, loaders["val"], config, device,
                               max_samples=attention_samples)
        _to_json(out_dir / "attention.json", attn)

    elapsed = round(time.time() - started, 1)
    cache_manifest = {}
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.exists():
        cache_manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
    config_payload = config.to_dict()
    config_payload.update({
        "seed": seed,
        "training_recipe": "v3.1",
        "best_epoch": best_epoch,
        "epochs_max": epochs,
        "patience": patience,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "effective_batch": batch_size * grad_accum,
        "lr": lr,
        "encoder_lr": encoder_lr,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "min_lr_ratio": min_lr_ratio,
        "amp": amp,
        "amp_active": bool(use_amp),
        "aux_weight": float(aux_weight),
        "aux_source": aux_source,
        "n_aux": (int(len(aux_loader.dataset))
                  if aux_loader is not None else 0),
        "device": str(device),
        "gpu": gpu_info(device),
        "n_train": int(len(split["train"])),
        "n_val": int(len(split["val"])),
        "n_test": int(len(split["test"])),
        "n_train_pos": int(labels[split["train"]].sum()),
        "n_val_pos": int(labels[split["val"]].sum()),
        "n_test_pos": int(labels[split["test"]].sum()),
        "threshold": float(threshold),
        "threshold_source": "validation set, max F1",
        "training_csv": str(ROOT / "data" / "training" / "confseq.csv"),
        "training_csv_sha256": sha256(
            ROOT / "data" / "training" / "confseq.csv"
        ),
        "cache_manifest": cache_manifest,
        "parameter_breakdown": breakdown,
        "n_params_total": int(total_params),
        "n_params_trainable": int(trainable_params),
        "elapsed_sec": elapsed,
        "code_sha256": code_manifest(),
        "label_definition": (
            "PPDB fungicide classification (1=fungicide list, "
            "0=herbicide/insecticide list); not experimental activity"
        ),
    })
    _to_json(out_dir / "config.json", config_payload)
    _to_json(out_dir / "metrics.json", {
        "train": {
            "loss": train_final["loss"],
            "roc_auc": train_final["roc_auc"],
            "pr_auc": train_final["pr_auc"],
        },
        "val": val_metrics,
        "test": test_metrics,
        "history": history,
        "attention_summary": _attention_summary(attn),
    })
    with (out_dir / "predictions_test.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["name", "smiles", "label", "probability",
                         "threshold", "predicted"])
        for idx, p in zip(split["test"], test_eval["probs"]):
            writer.writerow([
                rows[idx].get("name", ""), rows[idx]["smiles"],
                int(labels[idx]), f"{p:.6f}", f"{threshold:.6f}",
                int(p >= threshold),
            ])

    print(
        f"[V3.1/{'+'.join(config.modalities)}/{config.d_model}d/"
        f"{config.n_layers}L/seed{seed}] test AUC="
        f"{test_metrics['roc_auc']:.4f} "
        f"PR-AUC={test_metrics['pr_auc']:.4f} "
        f"F1={test_metrics['f1']:.4f} MCC={test_metrics['mcc']:.4f} "
        f"(best epoch {best_epoch}, {elapsed:.0f}s, {device})",
        flush=True,
    )
    return {
        "val": val_metrics,
        "test": test_metrics,
        "train": {"loss": train_final["loss"],
                  "roc_auc": train_final["roc_auc"],
                  "pr_auc": train_final["pr_auc"]},
        "history": history,
        "best_epoch": best_epoch,
        "threshold": float(threshold),
        "n_params_total": int(total_params),
        "n_params_trainable": int(trainable_params),
        "elapsed_sec": elapsed,
        "parameter_breakdown": breakdown,
        "attention": attn,
        "device": str(device),
    }


def _attention_summary(attn: dict | None) -> dict | None:
    if not attn:
        return None
    summary = {}
    for layer in attn["layers"]:
        block = layer["mean"]
        summary[f"layer{layer['layer']}"] = {
            src: {tgt: round(v["share"], 4) if v["share"] is not None else None
                  for tgt, v in dst.items()}
            for src, dst in block.items()
        }
    return summary


__all__ = [
    "train_v3_1", "resolve_device", "gpu_info", "parameter_breakdown",
    "gradient_group_norms", "attention_audit", "evaluate", "move_batch",
    "build_optimizer_groups",
]
