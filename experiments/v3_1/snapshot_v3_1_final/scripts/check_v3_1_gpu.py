"""实际验证 CUDA 与 V3.1 各模块是否在 GPU 上计算。

不是只看 torch.cuda.is_available()：会构建真实的 ChemBERTa + ConfSeq +
融合网络，跑前向/反向，检查每个模块的参数设备、混合精度和显存。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from molpredict.v3_1_train import (  # noqa: E402
    gpu_info,
    move_batch,
    nvidia_smi_status,
    resolve_device,
)
from molpredict.v3_data import (  # noqa: E402
    NumericStandardizer,
    V3Dataset,
    collate_v3,
    load_cache_arrays,
    load_rows,
)
from molpredict.v3_model import V3Config, V3Net  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


def _devices(module: torch.nn.Module) -> dict:
    out = {}
    for name, p in module.named_parameters():
        dev = str(p.device)
        out[dev] = out.get(dev, 0) + 1
    return out


def main() -> int:
    out_path = ROOT / "experiments" / "v3_1" / "gpu_check.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = resolve_device("cuda", require_cuda=True)
    report = {
        "gpu": gpu_info(device),
        "nvidia_smi": nvidia_smi_status(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "current_device": str(torch.cuda.current_device()),
    }

    mols, labels, rows = load_rows()
    arrays = load_cache_arrays(
        ROOT / "experiments" / "v2" / "cache",
        ROOT / "experiments" / "v3" / "cache",
        [r["inchikey"] for r in rows],
    )
    split = np.arange(8)
    standardizer = NumericStandardizer.fit(
        arrays["descriptors"], np.arange(len(rows))
    )
    ds = V3Dataset(
        split, labels,
        smiles_states=arrays["smiles_states"],
        smiles_ids=arrays["smiles_ids"],
        confseq_states=arrays["confseq_states"],
        confseq_ids=arrays["confseq_ids"],
        descriptors=standardizer.transform(arrays["descriptors"]),
    )
    loader = DataLoader(ds, batch_size=4, collate_fn=collate_v3)
    raw = next(iter(loader))

    config = V3Config(
        d_model=192, n_layers=6, n_heads=4,
        smiles_live=True, confseq_live=True,
        unfreeze_smiles_layers=2, unfreeze_confseq_layers=2,
    )
    model = V3Net(config).to(device)
    report["module_devices"] = {
        "smiles_encoder": _devices(model.smiles_encoder),
        "confseq_encoder": _devices(model.confseq_encoder),
        "smiles_projector": _devices(model.smiles_projector),
        "confseq_projector": _devices(model.confseq_projector),
        "descriptor_encoder": _devices(model.descriptor_encoder),
        "fusion": _devices(model.fusion),
        "classifier": _devices(model.classifier),
    }
    report["all_params_on_device"] = all(
        p.device.type == device.type
        and (device.index is None or p.device.index == device.index)
        for p in model.parameters()
    )

    batch = move_batch(raw, config, device)
    report["batch_devices"] = {
        k: (str(v[0].device) if isinstance(v, (tuple, list))
            else str(v.device))
        for k, v in batch.items()
    }

    # FP32 前向
    torch.cuda.reset_peak_memory_stats(device)
    model.eval()
    with torch.no_grad():
        logits = model(batch)
    report["forward_fp32"] = {
        "shape": list(logits.shape),
        "logits": logits.detach().float().cpu().tolist(),
        "peak_allocated_mb": round(
            torch.cuda.max_memory_allocated(device) / 1024 ** 2, 1
        ),
    }

    # BF16 AMP 前向 + 反向
    model.train()
    use_bf16 = torch.cuda.is_bf16_supported()
    report["bf16_supported"] = bool(use_bf16)
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=not use_bf16)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.time()
    with torch.autocast(device_type="cuda", dtype=dtype):
        logits = model(batch)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits.float(), batch["label"]
        )
    if scaler.is_enabled():
        scaler.scale(loss).backward()
    else:
        loss.backward()
    grads = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            key = name.split(".")[0]
            grads[key] = grads.get(key, 0.0) + float(
                p.grad.detach().float().pow(2).sum().item()
            )
    report["backward"] = {
        "loss": float(loss.item()),
        "amp_dtype": str(dtype),
        "grad_modules": {k: round(v ** 0.5, 6) for k, v in grads.items()},
        "elapsed_sec": round(time.time() - started, 3),
        "peak_allocated_mb": round(
            torch.cuda.max_memory_allocated(device) / 1024 ** 2, 1
        ),
        "peak_reserved_mb": round(
            torch.cuda.max_memory_reserved(device) / 1024 ** 2, 1
        ),
    }
    report["trainable_params"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    report["total_params"] = sum(p.numel() for p in model.parameters())
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
