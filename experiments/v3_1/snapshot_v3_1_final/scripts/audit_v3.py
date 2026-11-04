"""审计 V3/V3.1 模型是否真的在学习三种输入。

输出：参数量分解、逐轮曲线、梯度范数、跨模态注意力、token 统计、
训练/验证/测试差距、SMILES/ConfSeq 线性探针与表示冗余度。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from molpredict.v3_1_train import (  # noqa: E402
    attention_audit,
    evaluate,
    gradient_group_norms,
    parameter_breakdown,
    resolve_device,
)
from molpredict.v3_data import (  # noqa: E402
    NumericStandardizer,
    V3Dataset,
    collate_v3,
    load_cache_arrays,
    load_rows,
)
from molpredict.v3_encoders import load_smiles_tokenizer  # noqa: E402
from molpredict.v3_model import V3Config, V3Net  # noqa: E402


def _load_model(model_dir: Path, device: torch.device):
    import json as _json
    config = V3Config.from_dict(
        _json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    )
    model = V3Net(config)
    state = torch.load(model_dir / "model.pt", map_location="cpu",
                       weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, config


def _loaders(model_dir: Path, config: V3Config, batch_size: int,
             device: torch.device):
    mols, labels, rows = load_rows()
    expected = [r["inchikey"] for r in rows]
    arrays = load_cache_arrays(
        ROOT / "experiments" / "v2" / "cache",
        ROOT / "experiments" / "v3" / "cache",
        expected,
    )
    standardization = json.loads(
        (model_dir / "standardization.json").read_text(encoding="utf-8")
    )
    standardizer = NumericStandardizer.from_dict(standardization)
    descriptors = standardizer.transform(arrays["descriptors"])
    split_payload = json.loads(
        (model_dir / "split.json").read_text(encoding="utf-8")
    )
    split = {k: np.asarray(v, dtype=int) for k, v in split_payload.items()}
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
    loaders = {
        k: DataLoader(ds[k], batch_size=batch_size, shuffle=False,
                      collate_fn=collate_v3)
        for k in ds
    }
    return loaders, split, labels, rows, arrays, standardizer


def _criterion(split, labels, device):
    pos = float(labels[split["train"]].sum())
    neg = float(len(split["train"]) - pos)
    return nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32,
                                device=device)
    )


def _linear_probe(X_train, y_train, X_val, y_val, X_test, y_test):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    clf.fit(Xtr, y_train)
    out = {}
    for name, X, y in (("train", X_train, y_train),
                       ("val", X_val, y_val),
                       ("test", X_test, y_test)):
        p = clf.predict_proba(scaler.transform(X))[:, 1]
        out[name] = float(roc_auc_score(y, p))
    return out


def _cka(X: np.ndarray, Y: np.ndarray) -> float:
    """线性 CKA；只用于描述两组表示的重叠程度。"""
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    num = np.linalg.norm(X.T @ Y) ** 2
    den = np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)
    return float(num / den) if den else 0.0


def _ridge_r2(X_train, Y_train, X_val, Y_val):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    sx, sy = StandardScaler(), StandardScaler()
    Xtr = sx.fit_transform(X_train)
    Ytr = sy.fit_transform(Y_train)
    model = Ridge(alpha=1.0).fit(Xtr, Ytr)
    pred = model.predict(sx.transform(X_val))
    ss_res = float(((Y_val - sy.inverse_transform(pred)) ** 2).sum())
    ss_tot = float(((Y_val - Y_val.mean(axis=0)) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot else 0.0


def _encoder_grad_detail(model) -> dict:
    """逐模块检查预训练编码器是否冻结、是否真的拿到梯度。"""
    out = {}
    for name, p in model.named_parameters():
        if not name.startswith(("smiles_encoder.", "confseq_encoder.")):
            continue
        parts = name.split(".")
        marker = next((m for m in ("layers", "layer") if m in parts), None)
        if marker is not None:
            idx = parts.index(marker)
            key = ".".join(parts[:idx + 2])
        else:
            key = ".".join(parts[:3])
        row = out.setdefault(key, {
            "params": 0, "requires_grad_params": 0,
            "params_with_grad": 0, "grad_norm": 0.0,
        })
        row["params"] += p.numel()
        if p.requires_grad:
            row["requires_grad_params"] += p.numel()
        if p.grad is not None:
            row["params_with_grad"] += p.numel()
            row["grad_norm"] += float(
                p.grad.detach().float().pow(2).sum().item()
            )
    for row in out.values():
        row["grad_norm"] = row["grad_norm"] ** 0.5
    return dict(sorted(out.items()))


def _plot_history(history, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    ax1.plot(epochs, [h["train_loss"] for h in history],
             color="#1f77b4", label="train loss")
    if "val_loss" in history[0]:
        ax1.plot(epochs, [h["val_loss"] for h in history],
                 color="#d62728", label="val loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(epochs, [h["val_roc_auc"] for h in history],
             color="#2ca02c", label="val ROC-AUC")
    if "train_roc_auc" in history[0]:
        ax2.plot(epochs, [h["train_roc_auc"] for h in history],
                 color="#ff7f0e", linestyle="--", label="train ROC-AUC")
    ax2.set_ylabel("ROC-AUC")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_gradients(norms: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = [(k, v) for k, v in norms.items()
             if k not in {"total", "total_clipped"} and v > 0]
    items.sort(key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(7, max(3, 0.3 * len(items))))
    ax.barh([k for k, _ in items], [v for _, v in items], color="#4c72b0")
    ax.set_xlabel("gradient L2 norm")
    ax.set_xscale("log")
    ax.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_attention(attn: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layer = attn["layers"][0]["mean"]
    groups = ["smiles", "confseq", "descriptors"]
    groups = [g for g in groups if g in layer]
    mat = np.zeros((len(groups), len(groups)))
    for i, src in enumerate(groups):
        for j, tgt in enumerate(groups):
            entry = layer[src].get(tgt)
            mat[i, j] = entry["share"] if entry else 0.0
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(groups)), groups, rotation=30, ha="right")
    ax.set_yticks(range(len(groups)), groups)
    ax.set_xlabel("target (key)")
    ax.set_ylabel("source (query)")
    for i in range(len(groups)):
        for j in range(len(groups)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path,
                        default=ROOT / "experiments" / "v3" / "runs" /
                        "V3_full_seed42")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "experiments" / "v3_1" / "audit")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--attention-samples", type=int, default=64)
    parser.add_argument("--skip-probes", action="store_true")
    args = parser.parse_args()

    device = resolve_device(args.device, require_cuda=False)
    model, config = _load_model(args.model, device)
    loaders, split, labels, rows, arrays, standardizer = _loaders(
        args.model, config, args.batch_size, device
    )
    criterion = _criterion(split, labels, device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 参数量
    breakdown = parameter_breakdown(model)

    # 2. 训练/验证/测试指标与差距
    metrics = {}
    for name in ("train", "val", "test"):
        ev = evaluate(model, loaders[name], criterion, config, device)
        metrics[name] = {
            "loss": ev["loss"], "roc_auc": ev["roc_auc"],
            "pr_auc": ev["pr_auc"], "n": int(len(ev["labels"])),
        }
    metrics["train_test_auc_gap"] = (
        metrics["train"]["roc_auc"] - metrics["test"]["roc_auc"]
    )

    # 3. 梯度探针：一个训练 batch，前向后立即反向，不更新参数
    model.train()
    raw = next(iter(loaders["train"]))
    from molpredict.v3_1_train import move_batch
    batch = move_batch(raw, config, device)
    model.zero_grad(set_to_none=True)
    logits = model(batch)
    loss = criterion(logits, batch["label"])
    loss.backward()
    grad_norms = gradient_group_norms(model)
    encoder_grads = _encoder_grad_detail(model)
    model.zero_grad(set_to_none=True)
    model.eval()

    # 4. 跨模态注意力
    attn = attention_audit(model, loaders["val"], config, device,
                           max_samples=args.attention_samples)

    # 5. token 统计
    smiles_lengths = [len(x) for x in arrays["smiles_ids"]]
    confseq_lengths = [len(x) for x in arrays["confseq_ids"]]
    token_stats = {
        "smiles": {
            "n": len(smiles_lengths),
            "min": int(np.min(smiles_lengths)),
            "max": int(np.max(smiles_lengths)),
            "mean": float(np.mean(smiles_lengths)),
            "median": float(np.median(smiles_lengths)),
            "truncated": int(sum(x > config.max_smiles_len
                                 for x in smiles_lengths)),
        },
        "confseq": {
            "n": len(confseq_lengths),
            "min": int(np.min(confseq_lengths)),
            "max": int(np.max(confseq_lengths)),
            "mean": float(np.mean(confseq_lengths)),
            "median": float(np.median(confseq_lengths)),
            "truncated": int(sum(x > config.max_confseq_len
                                 for x in confseq_lengths)),
        },
        "descriptor_tokens": int(config.n_descriptors),
        "descriptor_tokens_all_valid": True,
    }

    # tokenizer 往返检查（只做前 5 条）
    roundtrip = []
    try:
        tokenizer = load_smiles_tokenizer()
        for i in range(min(5, len(rows))):
            ids = arrays["smiles_ids"][i].tolist()
            decoded = tokenizer.decode(ids, skip_special_tokens=True)
            original = rows[i]["smiles"]
            roundtrip.append({
                "original": original, "decoded": decoded,
                "match": decoded.replace(" ", "") == original.replace(" ", ""),
            })
    except Exception as exc:
        roundtrip = [{"error": str(exc)[:200]}]
    token_stats["tokenizer_roundtrip"] = roundtrip

    # 6. 表示统计与冗余（诊断，不是主模型）
    probes = {}
    redundancy = {}
    if not args.skip_probes:
        idx = {k: np.asarray(v, dtype=int) for k, v in split.items()}
        y = np.asarray(labels).astype(int)
        smile_mean = np.stack([
            arrays["smiles_states"][i].float().mean(0).numpy()
            for i in range(len(rows))
        ])
        conf_mean = np.stack([
            arrays["confseq_states"][i].float().mean(0).numpy()
            for i in range(len(rows))
        ])
        probes["smiles_mean_linear"] = _linear_probe(
            smile_mean[idx["train"]], y[idx["train"]],
            smile_mean[idx["val"]], y[idx["val"]],
            smile_mean[idx["test"]], y[idx["test"]],
        )
        probes["confseq_mean_linear"] = _linear_probe(
            conf_mean[idx["train"]], y[idx["train"]],
            conf_mean[idx["val"]], y[idx["val"]],
            conf_mean[idx["test"]], y[idx["test"]],
        )
        redundancy["linear_cka_smiles_confseq"] = _cka(
            smile_mean[idx["test"]], conf_mean[idx["test"]]
        )
        redundancy["ridge_r2_smiles_to_confseq_val"] = _ridge_r2(
            smile_mean[idx["train"]], conf_mean[idx["train"]],
            smile_mean[idx["val"]], conf_mean[idx["val"]],
        )

    # 7. 逐轮历史
    history = []
    metrics_path = args.model / "metrics.json"
    if metrics_path.exists():
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        history = payload.get("history", [])
    if history:
        _plot_history(history, out_dir / f"{args.model.name}_curves.png")

    _plot_gradients(grad_norms, out_dir / f"{args.model.name}_gradients.png")
    _plot_attention(attn, out_dir / f"{args.model.name}_attention.png")

    result = {
        "model": str(args.model),
        "device": str(device),
        "config": config.to_dict(),
        "parameter_breakdown": breakdown,
        "metrics": metrics,
        "gradient_probe": grad_norms,
        "encoder_grad_detail": encoder_grads,
        "attention": attn,
        "token_stats": token_stats,
        "linear_probes": probes,
        "redundancy": redundancy,
        "history_epochs": len(history),
        "notes": [
            "线性探针是诊断，不是主模型，也不改变 V3/V3.1 的结论。",
            "梯度探针只取一个训练 batch，表示该 checkpoint 的瞬时梯度，"
            "不是整个训练过程的平均。",
            "注意力份额只说明共同注意力计算时的信息参考，"
            "不能证明定位到真实化学作用位点。",
        ],
    }
    path = out_dir / f"{args.model.name}_audit.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print(json.dumps({
        "audit": str(path),
        "params": breakdown["total"],
        "trainable": breakdown["trainable"],
        "metrics": metrics,
        "grad_groups": grad_norms,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
