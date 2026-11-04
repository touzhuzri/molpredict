"""V3 模型/数据处理的关键正确性测试。"""

import numpy as np
import pytest
import torch
from torch import nn

from molpredict.padel import PADEL_NAMES
from molpredict.v3_data import NumericStandardizer
from molpredict.v3_encoders import set_trainable_last_layers
from molpredict.v3_model import V3Config, V3Net


def _config(**kwargs):
    base = dict(
        d_model=32, n_layers=1, n_heads=4, ff_mult=2, dropout=0.0,
        max_smiles_len=12, max_confseq_len=12,
    )
    base.update(kwargs)
    return V3Config(**base)


def _batch(batch_size=2, smiles_len=5, confseq_len=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    smiles_mask = torch.ones(batch_size, smiles_len, dtype=torch.bool)
    confseq_mask = torch.ones(batch_size, confseq_len, dtype=torch.bool)
    smiles_mask[1, 3:] = False
    confseq_mask[1, 2:] = False
    return {
        "smiles_states": torch.randn(batch_size, smiles_len, 768,
                                     generator=g),
        "smiles_mask": smiles_mask,
        "smiles_ids": (torch.ones(batch_size, smiles_len, dtype=torch.long),
                       smiles_mask),
        "confseq_states": torch.randn(batch_size, confseq_len, 256,
                                      generator=g),
        "confseq_mask": confseq_mask,
        "confseq_ids": (torch.ones(batch_size, confseq_len, dtype=torch.long),
                        confseq_mask),
        "descriptors": torch.randn(batch_size, 24, generator=g),
    }


def test_forward_shapes_and_variable_lengths():
    model = V3Net(_config()).eval()
    batch = _batch()
    with torch.no_grad():
        logits = model(batch)
    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()


def test_cls_mean_pooling_forward():
    model = V3Net(_config(pooling="cls_mean")).eval()
    batch = _batch()
    with torch.no_grad():
        logits = model(batch)
    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()


def test_padding_does_not_change_single_sample():
    """同一条样本在 batch 中补零后，结果应与单独输入一致。"""
    model = V3Net(_config()).eval()
    batch = _batch()
    with torch.no_grad():
        batched = model(batch)
        single = model({
            "smiles_states": batch["smiles_states"][1:2, :3],
            "smiles_mask": torch.ones(1, 3, dtype=torch.bool),
            "confseq_states": batch["confseq_states"][1:2, :2],
            "confseq_mask": torch.ones(1, 2, dtype=torch.bool),
            "descriptors": batch["descriptors"][1:2],
        })
    assert torch.allclose(batched[1:2], single, atol=1e-5)


@pytest.mark.parametrize("modalities", [
    ("smiles", "confseq", "descriptors"),
    ("smiles", "descriptors"),
    ("confseq", "descriptors"),
    ("smiles", "confseq"),
])
def test_modality_subsets(modalities):
    model = V3Net(_config(modalities=modalities)).eval()
    batch = _batch()
    with torch.no_grad():
        logits = model(batch)
    assert logits.shape == (2,)


def test_descriptor_tokens_are_separate_features():
    model = V3Net(_config()).eval()
    x = torch.zeros(2, 24)
    tokens = model.descriptor_encoder(x)
    assert tokens.shape == (2, 24, 32)
    # 数值为 0 时仍保留 24 个不同的身份 token
    assert not torch.allclose(tokens[0, 0], tokens[0, 1])


def test_descriptor_order_is_fixed():
    expected = [
        "nBr", "AATS7m", "ATSC0c", "AATSC5m", "AATSC6m", "VR1_DzZ",
        "nHdNH", "nHsSH", "nHsNH2", "nHsssNHp", "nHCsatu", "nssssNp",
        "naaO", "nssssSi", "nddssS", "SsBr", "minHBint8", "minHBint9",
        "n7Ring", "nG12Ring", "nF6Ring", "nF7Ring", "nF8Ring",
        "n6HeteroRing",
    ]
    assert PADEL_NAMES == expected


def test_standardizer_fits_train_only():
    X = np.array([[1.0], [np.nan], [1000.0]])
    s = NumericStandardizer.fit(X, np.array([0, 1]))
    # 中位数/均值/标准差只看训练行 [1.0, NaN→1.0]；1000 不参与
    assert s.median[0] == 1.0
    assert s.mean[0] == 1.0
    assert s.std[0] == 1.0
    transformed = s.transform(X)
    assert np.isfinite(transformed).all()


def test_freeze_last_layers():
    layers = nn.ModuleList([nn.Linear(2, 2) for _ in range(4)])
    set_trainable_last_layers(layers, 2)
    flags = [all(p.requires_grad for p in layer.parameters()) for layer in layers]
    assert flags == [False, False, True, True]
    assert any(p.requires_grad for p in layers[-1].parameters())
    assert all(not p.requires_grad for p in layers[0].parameters())


def test_backward_gradients():
    model = V3Net(_config())
    batch = _batch()
    loss = model(batch).sum()
    loss.backward()
    assert model.classifier[-1].weight.grad is not None
    assert model.descriptor_encoder.identity.weight.grad is not None
    assert torch.isfinite(model.classifier[-1].weight.grad).all()


def test_save_and_reload(tmp_path):
    model = V3Net(_config()).eval()
    batch = _batch()
    with torch.no_grad():
        before = model(batch)
    torch.save(model.state_dict(), tmp_path / "model.pt")
    reloaded = V3Net(_config()).eval()
    reloaded.load_state_dict(
        torch.load(tmp_path / "model.pt", weights_only=True), strict=True
    )
    with torch.no_grad():
        after = reloaded(batch)
    assert torch.allclose(before, after, atol=1e-6)


def test_repeated_prediction_is_deterministic():
    model = V3Net(_config()).eval()
    batch = _batch()
    with torch.no_grad():
        a = model(batch)
        b = model(batch)
    assert torch.equal(a, b)


def test_v2_scaffold_splits_are_disjoint():
    from molpredict.v3_data import load_split
    from molpredict.v3_data import V2_SPLITS

    if not V2_SPLITS.exists():
        pytest.skip("V2 splits not available")
    for seed in (42, 43, 44):
        split = load_split(seed)
        index_sets = [set(split[k].tolist()) for k in ("train", "val", "test")]
        assert not (index_sets[0] & index_sets[1])
        assert not (index_sets[0] & index_sets[2])
        assert not (index_sets[1] & index_sets[2])


def test_collate_padding_masks():
    from molpredict.v3_data import collate_v3

    batch = []
    for n in (3, 5):
        batch.append({
            "label": torch.tensor(1.0),
            "smiles_state": torch.randn(n, 8),
            "smiles_id": torch.arange(n),
            "confseq_state": torch.randn(n + 1, 8),
            "confseq_id": torch.arange(n + 1),
            "descriptors": torch.zeros(24),
        })
    out = collate_v3(batch)
    assert out["smiles_states"].shape == (2, 5, 8)
    assert out["smiles_mask"].sum().item() == 8
    assert out["confseq_states"].shape == (2, 6, 8)
    assert out["confseq_mask"].sum().item() == 10
    assert out["descriptors"].shape == (2, 24)
