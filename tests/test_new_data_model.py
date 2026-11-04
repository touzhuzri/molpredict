"""新数据辅助模型的最小测试。"""
from __future__ import annotations
import torch
from molpredict.new_data_model import AuxDataset, AuxV3Net, collate_aux
from molpredict.v3_model import V3Config


def _config():
    return V3Config(modalities=("smiles", "descriptors"), d_model=16,
                    n_layers=2, n_heads=2, ff_mult=2, dropout=0.0,
                    max_smiles_len=5, n_descriptors=3, smiles_dim=8,
                    descriptor_value_hidden=8)


def _batch():
    return {
        "label": torch.tensor([1.0, 0.0]),
        "smiles_states": torch.randn(2, 5, 8),
        "smiles_mask": torch.tensor([[1, 1, 1, 0, 0],
                                     [1, 1, 1, 1, 1]], dtype=torch.bool),
        "descriptors": torch.randn(2, 3),
    }


def test_aux_model_main_and_aux_shapes():
    model = AuxV3Net(_config(), aux_dim=3, aux_hidden=8)
    assert model(_batch()).shape == (2,)
    aux = model.forward_aux(_batch())
    assert aux.shape == (2, 3)
    loss = aux.pow(2).mean() + model(_batch()).pow(2).mean()
    loss.backward()
    assert model.aux_head[1].weight.grad is not None
    assert model.backbone.smiles_projector.proj[0].weight.grad is not None


def test_aux_collate_masks_padding():
    ds = AuxDataset(
        [torch.randn(3, 8), torch.randn(5, 8)],
        [[1.0, 0.0, 1.0], [0.5, 0.5, 0.5]],
        [[1, 1, 1], [1, 0, 1]],
    )
    batch = collate_aux([ds[0], ds[1]])
    assert batch["smiles_states"].shape == (2, 5, 8)
    assert batch["smiles_mask"].tolist()[0] == [True, True, True, False, False]
    assert batch["aux_target"].shape == (2, 3)
    assert batch["aux_mask"].tolist()[1] == [1.0, 0.0, 1.0]
