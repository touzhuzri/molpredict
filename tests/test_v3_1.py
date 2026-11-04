"""V3.1 训练与诊断组件的最小测试。"""

from __future__ import annotations

import math

import torch
from torch import nn

from molpredict.v3_1_train import (
    _attention_weights,
    build_optimizer_groups,
    gradient_group_norms,
    parameter_breakdown,
    warmup_cosine,
)
from molpredict.v3_model import V3Config, V3Net


def _tiny_config(**kwargs):
    base = dict(
        d_model=16, n_layers=2, n_heads=2, ff_mult=2, dropout=0.0,
        max_smiles_len=6, max_confseq_len=5, n_descriptors=3,
        smiles_dim=8, confseq_dim=8, descriptor_value_hidden=8,
    )
    base.update(kwargs)
    return V3Config(**base)


def _batch():
    bsz = 2
    smiles = torch.randn(bsz, 6, 8)
    smiles_mask = torch.zeros(bsz, 6, dtype=torch.bool)
    smiles_mask[0, :4] = True
    smiles_mask[1, :6] = True
    confseq = torch.randn(bsz, 5, 8)
    confseq_mask = torch.zeros(bsz, 5, dtype=torch.bool)
    confseq_mask[0, :3] = True
    confseq_mask[1, :5] = True
    return {
        "label": torch.tensor([1.0, 0.0]),
        "smiles_states": smiles,
        "smiles_mask": smiles_mask,
        "confseq_states": confseq,
        "confseq_mask": confseq_mask,
        "descriptors": torch.randn(bsz, 3),
    }


def test_embed_layout_and_mask():
    model = V3Net(_tiny_config())
    x, mask = model.embed(_batch())
    # 1 CLS + 6 SMILES + 5 ConfSeq + 3 descriptors
    assert x.shape == (2, 15, 16)
    assert mask.shape == (2, 15)
    assert mask[:, 0].all()  # CLS 始终有效
    assert mask[:, -3:].all()  # 24（此处 3）个 descriptor 全部有效
    assert int(mask[0].sum()) == 1 + 4 + 3 + 3
    assert int(mask[1].sum()) == 1 + 6 + 5 + 3


def test_embed_padding_is_masked_out():
    model = V3Net(_tiny_config())
    x, mask = model.embed(_batch())
    # 被 mask 的 token 仍然有 modality embedding，但 key-padding mask 会排除它们；
    # 这里验证 mask 本身与长度一致，而不是要求表示为零。
    assert not mask[0, 1 + 4:1 + 6].any()
    assert not mask[0, 1 + 6 + 3:1 + 6 + 5].any()


def test_attention_weights_masked_and_normalized():
    torch.manual_seed(0)
    mha = nn.MultiheadAttention(16, 2, batch_first=True)
    x = torch.randn(2, 7, 16)
    key_padding = torch.zeros(2, 7, dtype=torch.bool)
    key_padding[0, 5:] = True
    key_padding[1, 6] = True
    weights = _attention_weights(x, mha, key_padding)
    assert weights.shape == (2, 2, 7, 7)
    # 每个 query 在有效 key 上的权重和为 1
    valid = (~key_padding).unsqueeze(1).unsqueeze(2)
    sums = (weights * valid).sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    # 被 mask 的 key 权重为 0
    assert torch.count_nonzero(weights[0, :, :, 5:]) == 0


def test_parameter_breakdown_and_grad_groups():
    model = V3Net(_tiny_config())
    breakdown = parameter_breakdown(model)
    assert breakdown["total"] == breakdown["trainable"]
    assert breakdown["total"] > 0
    assert "smiles_projector" in breakdown["modules"]
    assert "descriptor_encoder" in breakdown["modules"]
    assert "fusion_layer_0" in breakdown["modules"]

    model.train()
    logits = model(_batch())
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, torch.tensor([1.0, 0.0])
    )
    loss.backward()
    norms = gradient_group_norms(model)
    assert norms["total"] > 0
    for key in ("smiles_projector", "confseq_projector",
                "descriptor_encoder", "classifier"):
        assert key in norms and norms[key] > 0


def test_frozen_encoder_has_no_gradient_and_groups_split():
    model = V3Net(_tiny_config())
    # 没有 live 编码器时只有融合组
    optimizer = build_optimizer_groups(model, 1e-3, 2e-5, 1e-4)
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["name"] == "fusion"


def test_warmup_cosine_shape():
    param = nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([param], lr=1e-3)
    sched = warmup_cosine(opt, total_steps=10, warmup_ratio=0.2,
                          min_lr_ratio=0.1)
    lrs = []
    for _ in range(10):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] < lrs[1]  # warmup
    assert lrs[-1] < max(lrs)  # 余弦衰减
    assert lrs[-1] >= 1e-3 * 0.1 - 1e-12
    assert math.isclose(max(lrs), 1e-3, rel_tol=1e-6)


def test_arch_presets_are_divisible():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / \
        "run_v3_1_experiments.py"
    spec = importlib.util.spec_from_file_location("run_v31", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name, (d_model, n_layers, n_heads) in module.ARCHS.items():
        assert d_model % n_heads == 0, name
        assert n_layers > 0 and n_heads > 0


def test_attention_auditor_collects_cross_modal_mass():
    from molpredict.v3_1_train import AttentionAuditor

    model = V3Net(_tiny_config())
    batch = _batch()
    x, mask = model.embed(batch)
    auditor = AttentionAuditor()
    auditor.attach(model)
    positions = []
    for b in range(2):
        positions.append({
            "smiles": list(range(1, 1 + int(batch["smiles_mask"][b].sum()))),
            "confseq": list(range(7, 7 + int(
                batch["confseq_mask"][b].sum()))),
            "descriptors": list(range(12, 15)),
        })
    auditor.set_context(positions, mask)
    with torch.no_grad():
        model.fusion(x, src_key_padding_mask=~mask)
    auditor.detach()
    result = auditor.result({0: 2, 1: 2})
    assert result["n_samples"] == 2
    layer0 = result["layers"][0]["mean"]
    assert "smiles" in layer0 and "confseq" in layer0["smiles"]
    share = layer0["smiles"]["confseq"]["share"]
    assert 0.0 <= share <= 1.0
