"""V3 预训练编码器封装：冻结 token 表示、可选部分解冻、缓存。

V3 的核心模型在 ``v3_model.py``；本模块只负责把 SMILES / ConfSeq 变成
token 级隐藏表示，并明确区分“缓存冻结表示”和“可训练编码器”。
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from molpredict.v2 import (
    CONFSEQ_CHECKPOINT,
    CONFSEQ_ROOT,
    SMILES_MODEL,
)

# V3 不再沿用 V2 的 128 token 截断：当前数据未截断最大 298，
# 320 可以完整覆盖；ConfSeq 最大 499，512 完整覆盖。
MAX_SMILES_LEN = 320
MAX_CONFSEQ_LEN = 512
SMILES_PAD_ID = 1
CONFSEQ_PAD_ID = 458
CONFSEQ_UNK_ID = 3


def set_trainable_last_layers(layers: nn.ModuleList, n: int) -> None:
    """先冻结所有层，再只解冻最后 n 层；n<=0 表示全部冻结。"""
    for layer in layers:
        for p in layer.parameters():
            p.requires_grad = False
    if n > 0:
        for layer in layers[-n:]:
            for p in layer.parameters():
                p.requires_grad = True


def load_smiles_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(SMILES_MODEL, local_files_only=True)


def load_smiles_encoder(unfreeze_last: int = 0):
    """加载 ChemBERTa 基础编码器；默认全部冻结。"""
    from transformers import AutoModel

    model = AutoModel.from_pretrained(SMILES_MODEL, local_files_only=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    if unfreeze_last:
        set_trainable_last_layers(model.encoder.layer, unfreeze_last)
    return model


def load_confseq_parts(unfreeze_last: int = 0):
    """加载官方 ConfSeq 编码器、词表和特殊 token id。

    只调用官方 ``get_confseq_embeddings`` 的构建/加载函数，不改其算法。
    """
    sys.path.insert(0, str(CONFSEQ_ROOT / "representation_learning"))
    from get_confseq_embeddings import (  # type: ignore
        build_bart_encoder,
        build_confseq_vocab,
        load_checkpoint_to_encoder,
    )

    vocab, vocab_dict = build_confseq_vocab()
    pad_id = int(vocab_dict["<pad>"])
    unk_id = int(vocab_dict["<unk>"])
    encoder = build_bart_encoder(
        vocab_size=len(vocab),
        pad_id=pad_id,
        eos_id=vocab_dict["<eos>"],
        sos_id=vocab_dict["<sos>"],
    )
    load_checkpoint_to_encoder(encoder, str(CONFSEQ_CHECKPOINT))
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    if unfreeze_last:
        set_trainable_last_layers(
            encoder.bart_model.model.encoder.layers, unfreeze_last
        )
    return encoder, vocab_dict, pad_id, unk_id


def tokenize_confseq(seqs, vocab_dict, max_len: int = MAX_CONFSEQ_LEN,
                     unk_id: int = CONFSEQ_UNK_ID):
    """官方 ConfSeq 输入是空格分隔的离散 token；按空格切分并查词表。"""
    ids = []
    for seq in seqs:
        tokens = str(seq).split()
        if not tokens:
            raise ValueError("ConfSeq 字符串为空")
        if len(tokens) > max_len:
            raise ValueError(f"ConfSeq 长度 {len(tokens)} 超过上限 {max_len}")
        ids.append(torch.tensor(
            [vocab_dict.get(t, unk_id) for t in tokens],
            dtype=torch.long,
        ))
    return ids


def encode_smiles_states(smiles, tokenizer, encoder, max_length=MAX_SMILES_LEN,
                         batch_size=64):
    """返回每条 SMILES 的 token 级 hidden state（不池化）和 token id/mask。"""
    states, ids_list, masks, truncated = [], [], [], 0
    encoder.eval()
    with torch.no_grad():
        for start in range(0, len(smiles), batch_size):
            chunk = smiles[start:start + batch_size]
            enc = tokenizer(
                chunk, padding=True, truncation=True, max_length=max_length,
                return_tensors="pt",
            )
            hidden = encoder(**enc).last_hidden_state
            for j in range(len(chunk)):
                length = int(enc["attention_mask"][j].sum())
                n_raw = len(tokenizer(chunk[j], truncation=False)["input_ids"])
                if n_raw > max_length:
                    truncated += 1
                states.append(hidden[j, :length].to(torch.float16).cpu())
                ids_list.append(enc["input_ids"][j, :length].cpu())
                masks.append(enc["attention_mask"][j, :length].cpu())
    return states, ids_list, masks, truncated


def encode_confseq_states(seqs, encoder, vocab_dict, pad_id=CONFSEQ_PAD_ID,
                          unk_id=CONFSEQ_UNK_ID,
                          max_len=MAX_CONFSEQ_LEN, batch_size=64):
    """返回官方 ConfSeq 编码器的 token 级 hidden state，不池化。"""
    ids = tokenize_confseq(seqs, vocab_dict, max_len=max_len, unk_id=unk_id)
    states, ids_list, masks = [], [], []
    encoder.eval()
    with torch.no_grad():
        for start in range(0, len(ids), batch_size):
            chunk = ids[start:start + batch_size]
            padded = torch.nn.utils.rnn.pad_sequence(
                chunk, batch_first=True, padding_value=pad_id
            )
            mask = (padded != pad_id).long()
            hidden = encoder(input_ids=padded, attention_mask=mask)
            for j in range(len(chunk)):
                length = int(mask[j].sum())
                states.append(hidden[j, :length].to(torch.float16).cpu())
                ids_list.append(padded[j, :length].cpu())
                masks.append(mask[j, :length].cpu())
    return states, ids_list, masks


def cache_smiles_states(rows, cache_dir: Path, max_length=MAX_SMILES_LEN) -> Path:
    """生成 V3 的 SMILES token 缓存（覆盖 V2 的 128 截断）。"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / "smiles_states.pt"
    tokens_out = cache_dir / "smiles_tokens.pt"
    meta_out = cache_dir / "smiles_meta.json"
    if out.exists() and tokens_out.exists() and meta_out.exists():
        try:
            meta = json.loads(meta_out.read_text(encoding="utf-8"))
            if meta.get("max_length") == max_length:
                return out
        except Exception:
            pass
    print(f"缓存 V3 ChemBERTa token 表示（max_length={max_length}）...")
    tokenizer = load_smiles_tokenizer()
    encoder = load_smiles_encoder(0)
    smiles = [r["smiles"] for r in rows]
    started = time.time()
    states, ids, masks, truncated = encode_smiles_states(
        smiles, tokenizer, encoder, max_length=max_length
    )
    torch.save({"states": states, "inchikeys": [r["inchikey"] for r in rows]}, out)
    torch.save({"input_ids": ids, "attention_mask": masks,
                "inchikeys": [r["inchikey"] for r in rows]}, tokens_out)
    meta = {
        "model": SMILES_MODEL,
        "max_length": max_length,
        "n": len(states),
        "max_observed": int(max(len(x) for x in ids)),
        "truncated": int(truncated),
        "elapsed_sec": round(time.time() - started, 1),
    }
    meta_out.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"V3 SMILES 缓存完成: {meta}")
    return out


def cache_confseq_states(rows, cache_dir: Path) -> Path:
    """生成官方 ConfSeq token 缓存（若不存在）；不修改官方编码器。"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / "confseq_states.pt"
    tokens_out = cache_dir / "confseq_tokens.pt"
    if out.exists() and tokens_out.exists():
        return out
    print("缓存官方 ConfSeq token 表示...")
    encoder, vocab_dict, pad_id, unk_id = load_confseq_parts(0)
    seqs = [r["confseq"] for r in rows]
    started = time.time()
    states, ids, masks = encode_confseq_states(
        seqs, encoder, vocab_dict, pad_id=pad_id, unk_id=unk_id
    )
    torch.save({"states": states, "inchikeys": [r["inchikey"] for r in rows]}, out)
    torch.save({"input_ids": ids, "attention_mask": masks,
                "inchikeys": [r["inchikey"] for r in rows]}, tokens_out)
    meta = {
        "checkpoint": str(CONFSEQ_CHECKPOINT),
        "checkpoint_sha256": sha256(CONFSEQ_CHECKPOINT),
        "max_length": MAX_CONFSEQ_LEN,
        "n": len(states),
        "max_observed": int(max(len(x) for x in ids)),
        "elapsed_sec": round(time.time() - started, 1),
    }
    (cache_dir / "confseq_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"V3 ConfSeq 缓存完成: {meta}")
    return out


def save_tokenizer(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    load_smiles_tokenizer().save_pretrained(str(out_dir))


def save_confseq_vocab(out_dir: Path) -> None:
    _, vocab_dict, pad_id, unk_id = load_confseq_parts(0)
    payload = {
        "vocab": vocab_dict,
        "pad_id": pad_id,
        "unk_id": unk_id,
        "checkpoint": str(CONFSEQ_CHECKPOINT),
        "checkpoint_sha256": sha256(CONFSEQ_CHECKPOINT),
        "source": "official ConfSeq representation_learning",
    }
    (out_dir / "confseq_vocab.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pretrained_manifest() -> dict:
    """记录 V3 使用的预训练来源；不声称未核实的许可证。"""
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(SMILES_MODEL, local_files_only=True)
    return {
        "smiles": {
            "name": SMILES_MODEL,
            "hidden_size": int(cfg.hidden_size),
            "layers": int(cfg.num_hidden_layers),
            "source": "Hugging Face cache; see README / model card",
            "license": "代码仓库 MIT；模型权重许可证未在本地 model card 中单独确认",
        },
        "confseq": {
            "checkpoint": str(CONFSEQ_CHECKPOINT),
            "sha256": sha256(CONFSEQ_CHECKPOINT),
            "source": "official ConfSeq code + checkpoint (HF mirror Oopstom/confseq-shape-screen)",
            "license": "ConfSeq repository AGPL-3.0; weight redistribution needs confirmation",
        },
        "padel": {
            "tool": "PaDEL-Descriptor via padelpy 0.1.17",
            "license": "jar AGPL-3.0",
        },
    }
