"""V2 多模态杀菌剂分类：SMILES + ConfSeq + 数值描述符。"""

import csv
import json
import os
import random
import sys
import time
from pathlib import Path

# 先设置 HF 镜像，再导入 transformers
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from rdkit import Chem
from rdkit.Chem import Descriptors

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit

from molpredict.modeling import (
    descriptor_features,
    load_dataset,
    make_split,
    morgan_features,
    scaffold_groups,
)

ROOT = Path(__file__).resolve().parents[2]
CONFSEQ_ROOT = ROOT / "data" / "external" / "ConfSeq"
DEFAULT_CACHE = ROOT / "experiments" / "v2" / "cache"
DEFAULT_RUNS = ROOT / "experiments" / "v2" / "runs"
DEFAULT_EXPERIMENTS = Path("experiments/v2")
SMILES_MODEL = "seyonec/ChemBERTa-zinc-base-v1"
CONFSEQ_CHECKPOINT = (
    CONFSEQ_ROOT / "representation_learning" / "checkpoints" / "model_epoch_1.pth"
)

EXPERIMENTS = {
    "B": {"modalities": ["smiles"], "numeric": None},
    "C": {"modalities": ["confseq"], "numeric": None},
    "D": {"modalities": ["numeric"], "numeric": "padel24"},
    "E": {"modalities": ["smiles", "confseq"], "numeric": None},
    "F": {"modalities": ["smiles", "confseq", "numeric"], "numeric": "padel24"},
    "G": {"modalities": ["smiles", "confseq", "numeric"], "numeric": "rdkit217"},
}


# ---------------------------------------------------------------------------
# 特征缓存
# ---------------------------------------------------------------------------
def _save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_rows(csv_path: Path):
    mols, labels, rows = load_dataset(csv_path)
    return mols, labels, rows


def cache_smiles(rows, cache_dir: Path) -> None:
    states_path = cache_dir / "smiles_states.pt"
    tokens_path = cache_dir / "smiles_tokens.pt"
    if states_path.exists() and tokens_path.exists():
        return
    print("缓存 ChemBERTa token 表示...")
    tokenizer = AutoTokenizer.from_pretrained(SMILES_MODEL)
    model = AutoModel.from_pretrained(SMILES_MODEL)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    smiles = [r["smiles"] for r in rows]
    states, input_ids, attention_masks = [], [], []
    batch_size = 64
    started = time.time()
    with torch.no_grad():
        for start in range(0, len(smiles), batch_size):
            chunk = smiles[start:start + batch_size]
            enc = tokenizer(
                chunk, padding=True, truncation=True, max_length=128,
                return_tensors="pt",
            )
            hidden = model(**enc).last_hidden_state
            for j in range(len(chunk)):
                length = int(enc["attention_mask"][j].sum())
                states.append(hidden[j, :length].to(torch.float16).cpu())
                input_ids.append(enc["input_ids"][j, :length].cpu())
                attention_masks.append(enc["attention_mask"][j, :length].cpu())
    torch.save({"states": states, "inchikeys": [r["inchikey"] for r in rows]},
               states_path)
    torch.save({"input_ids": input_ids, "attention_mask": attention_masks,
                "inchikeys": [r["inchikey"] for r in rows]}, tokens_path)
    print(f"ChemBERTa 缓存完成: {len(states)} 条，{time.time()-started:.1f}s，"
          f"max_len={max(len(x) for x in states)}")


def _confseq_encoder():
    sys.path.insert(0, str(CONFSEQ_ROOT / "representation_learning"))
    from get_confseq_embeddings import (
        build_bart_encoder,
        build_confseq_vocab,
        load_checkpoint_to_encoder,
    )
    vocab, vocab_dict = build_confseq_vocab()
    pad_id = vocab_dict["<pad>"]
    unk_id = vocab_dict["<unk>"]
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
    return encoder, vocab_dict, pad_id, unk_id


def cache_confseq(rows, cache_dir: Path) -> None:
    states_path = cache_dir / "confseq_states.pt"
    tokens_path = cache_dir / "confseq_tokens.pt"
    if states_path.exists() and tokens_path.exists():
        return
    print("缓存官方 ConfSeq 编码器 token 表示...")
    encoder, vocab_dict, pad_id, unk_id = _confseq_encoder()
    seqs = [r["confseq"] for r in rows]
    lengths = [len(s.split()) for s in seqs]
    print(f"ConfSeq token 长度: max={max(lengths)}, mean={np.mean(lengths):.1f}")
    if max(lengths) > 1024:
        raise ValueError("ConfSeq 序列超过 1024 token，需要确认官方模型位置编码后再处理")

    states, ids_list, masks = [], [], []
    batch_size = 64
    started = time.time()
    with torch.no_grad():
        for start in range(0, len(seqs), batch_size):
            chunk = seqs[start:start + batch_size]
            ids = [torch.tensor(
                [vocab_dict.get(t, unk_id) for t in s.split()], dtype=torch.long
            ) for s in chunk]
            padded = pad_sequence(ids, batch_first=True, padding_value=pad_id)
            mask = (padded != pad_id).long()
            hidden = encoder(input_ids=padded, attention_mask=mask)
            for j in range(len(chunk)):
                length = int(mask[j].sum())
                states.append(hidden[j, :length].to(torch.float16).cpu())
                ids_list.append(padded[j, :length].cpu())
                masks.append(mask[j, :length].cpu())
    torch.save({"states": states, "inchikeys": [r["inchikey"] for r in rows]},
               states_path)
    torch.save({"input_ids": ids_list, "attention_mask": masks,
                "inchikeys": [r["inchikey"] for r in rows]}, tokens_path)
    print(f"ConfSeq 缓存完成: {len(states)} 条，{time.time()-started:.1f}s")


def cache_numeric(rows, cache_dir: Path) -> None:
    from molpredict.padel import compute_padel24

    padel_csv = compute_padel24([r["smiles"] for r in rows], cache_dir)
    rdkit_path = cache_dir / "rdkit217.npy"
    if not rdkit_path.exists():
        print("缓存 RDKit 217 描述符...")
        X = np.vstack([descriptor_features(Chem.MolFromSmiles(r["smiles"]))
                       for r in rows])
        np.save(rdkit_path, X.astype(np.float32))
        _save_json(cache_dir / "rdkit217_meta.json", {
            "source": "RDKit Descriptors.descList",
            "n_descriptors": int(X.shape[1]),
            "names": [n for n, _ in Descriptors.descList],
        })
    _save_json(cache_dir / "padel24_order.json", {
        "padel_csv": str(padel_csv),
        "rdkit_npy": str(rdkit_path),
        "inchikeys": [r["inchikey"] for r in rows],
    })


def prepare_cache(csv_path: Path, cache_dir: Path = DEFAULT_CACHE) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    _, _, rows = _load_rows(csv_path)
    cache_smiles(rows, cache_dir)
    cache_confseq(rows, cache_dir)
    cache_numeric(rows, cache_dir)
    print(f"缓存目录: {cache_dir}")


# ---------------------------------------------------------------------------
# 划分
# ---------------------------------------------------------------------------
def three_way_split(mols, labels, seed: int) -> dict:
    train_idx = test_idx = None
    for outer in range(10):
        train_idx, test_idx = make_split(mols, labels, seed=seed + outer * 100)
        if len(np.unique(labels[test_idx])) == 2:
            break
    else:
        raise ValueError("无法得到含两类的测试集")

    groups = scaffold_groups(mols)
    train_groups = [groups[i] for i in train_idx]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    for attempt in range(10):
        rel_train, rel_val = next(splitter.split(
            train_idx, labels[train_idx], train_groups
        ))
        tr = train_idx[rel_train]
        va = train_idx[rel_val]
        if len(np.unique(labels[tr])) == 2 and len(np.unique(labels[va])) == 2:
            return {"train": tr, "val": va, "test": test_idx}
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.15,
                                     random_state=seed + attempt + 1)
    raise ValueError("无法得到含两类的验证集")


def save_splits(mols, labels, seeds, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = {}
    for seed in seeds:
        split = three_way_split(mols, labels, seed)
        splits[str(seed)] = {k: v.tolist() for k, v in split.items()}
        splits[str(seed)]["counts"] = {
            k: {"n": int(len(v)), "pos": int(labels[v].sum())}
            for k, v in split.items()
        }
        _save_json(out_dir / f"split_seed{seed}.json", splits[str(seed)])
        print(f"split {seed}: train={len(split['train'])}, "
              f"val={len(split['val'])}, test={len(split['test'])}")
    return splits


# ---------------------------------------------------------------------------
# 数据集 / collate
# ---------------------------------------------------------------------------
class V2Dataset(Dataset):
    def __init__(self, indices, labels, *, numeric=None, smiles_states=None,
                 confseq_states=None, smiles_ids=None):
        self.indices = np.asarray(indices)
        self.labels = labels
        self.numeric = numeric
        self.smiles_states = smiles_states
        self.confseq_states = confseq_states
        self.smiles_ids = smiles_ids

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        item = {"label": torch.tensor(float(self.labels[idx]), dtype=torch.float32)}
        if self.numeric is not None:
            item["numeric"] = torch.tensor(self.numeric[idx], dtype=torch.float32)
        if self.smiles_states is not None:
            item["smiles"] = self.smiles_states[idx].float()
        if self.confseq_states is not None:
            item["confseq"] = self.confseq_states[idx].float()
        if self.smiles_ids is not None:
            item["smiles_ids"] = self.smiles_ids[idx]
        return item


def collate_v2(batch):
    out = {"label": torch.stack([b["label"] for b in batch])}
    if "numeric" in batch[0]:
        out["numeric"] = torch.stack([b["numeric"] for b in batch])
    for key in ("smiles", "confseq"):
        if key not in batch[0]:
            continue
        seqs = [b[key] for b in batch]
        mask = pad_sequence(
            [torch.ones(s.size(0), dtype=torch.bool) for s in seqs],
            batch_first=True, padding_value=False,
        )
        out[key] = (pad_sequence(seqs, batch_first=True, padding_value=0.0), mask)
    if "smiles_ids" in batch[0]:
        ids = [b["smiles_ids"] for b in batch]
        mask = pad_sequence(
            [torch.ones(s.size(0), dtype=torch.bool) for s in ids],
            batch_first=True, padding_value=False,
        )
        pad_id = 1
        out["smiles_ids"] = (
            pad_sequence(ids, batch_first=True, padding_value=pad_id), mask
        )
    return out


# ---------------------------------------------------------------------------
# 网络
# ---------------------------------------------------------------------------
class AttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Linear(dim, 1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, mask):
        logits = self.score(x).squeeze(-1)
        logits = logits.masked_fill(~mask, -1e9)
        weights = torch.softmax(logits, dim=1)
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return self.norm(pooled)


class TokenBranch(nn.Module):
    def __init__(self, in_dim, dim, dropout):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.pool = AttentionPool(dim)

    def forward(self, x, mask):
        return self.pool(self.proj(x), mask)


class NumericBranch(nn.Module):
    def __init__(self, in_dim, dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class GatedFusion(nn.Module):
    def __init__(self, dim, n_modalities, dropout):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * n_modalities, dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim, n_modalities),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, zs):
        gates = torch.softmax(self.gate(torch.cat(zs, dim=1)), dim=1)
        fused = sum(gates[:, i:i + 1] * zs[i] for i in range(len(zs)))
        return self.norm(fused), gates


class AttentionFusion(nn.Module):
    def __init__(self, dim, n_modalities, dropout, heads=4):
        super().__init__()
        self.type_embed = nn.Parameter(torch.zeros(n_modalities, dim))
        self.cls = nn.Parameter(torch.zeros(1, dim))
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 2,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, zs):
        tokens = torch.stack(zs, dim=1) + self.type_embed
        cls = self.cls.expand(zs[0].size(0), -1, -1)
        fused = self.encoder(torch.cat([cls, tokens], dim=1))
        return self.norm(fused[:, 0]), None


class LiveSmilesBranch(nn.Module):
    """用于部分解冻微调：编码器在模型内部，token ids 随 batch 输入。"""
    def __init__(self, dim, dropout, unfreeze_layers):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(SMILES_MODEL)
        for p in self.encoder.parameters():
            p.requires_grad = False
        if unfreeze_layers:
            for layer in self.encoder.encoder.layer[-unfreeze_layers:]:
                for p in layer.parameters():
                    p.requires_grad = True
        self.proj = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, dim),
            nn.LayerNorm(dim), nn.GELU(), nn.Dropout(dropout),
        )
        self.pool = AttentionPool(dim)

    def forward(self, input_ids, mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=mask.long()).last_hidden_state
        return self.pool(self.proj(hidden), mask)


class V2Net(nn.Module):
    def __init__(self, modalities, numeric_dim, dim=128, fusion="attention",
                 dropout=0.2, live_smiles=False, unfreeze_layers=2):
        super().__init__()
        self.modalities = list(modalities)
        if "smiles" in modalities:
            if live_smiles:
                self.smiles_branch = LiveSmilesBranch(dim, dropout, unfreeze_layers)
            else:
                self.smiles_branch = TokenBranch(768, dim, dropout)
        if "confseq" in modalities:
            self.confseq_branch = TokenBranch(256, dim, dropout)
        if "numeric" in modalities:
            self.numeric_branch = NumericBranch(numeric_dim, dim, dropout)
        self.fusion = None
        if len(modalities) > 1:
            if fusion == "gated":
                self.fusion = GatedFusion(dim, len(modalities), dropout)
            elif fusion == "attention":
                self.fusion = AttentionFusion(dim, len(modalities), dropout)
            else:
                raise ValueError(f"未知融合方式: {fusion}")
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim // 2, 1),
        )

    def forward(self, batch):
        zs = []
        for modality in self.modalities:
            if modality == "smiles":
                if "smiles_ids" in batch:
                    ids, mask = batch["smiles_ids"]
                    zs.append(self.smiles_branch(ids, mask))
                else:
                    x, mask = batch["smiles"]
                    zs.append(self.smiles_branch(x, mask))
            elif modality == "confseq":
                x, mask = batch["confseq"]
                zs.append(self.confseq_branch(x, mask))
            else:
                zs.append(self.numeric_branch(batch["numeric"]))
        if self.fusion is not None:
            fused, _ = self.fusion(zs)
        else:
            fused = zs[0]
        return self.classifier(fused).squeeze(-1)


# ---------------------------------------------------------------------------
# 评价
# ---------------------------------------------------------------------------
def binary_metrics(y_true, proba, threshold=0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(proba) >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    return {
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "confusion_matrix": {
            "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
        },
    }


def best_threshold(y_true, proba) -> float:
    thresholds = np.unique(np.round(proba, 4))
    best, best_f1 = 0.5, -1.0
    for t in thresholds:
        score = f1_score(y_true, (proba >= t).astype(int), zero_division=0)
        if score > best_f1:
            best, best_f1 = float(t), score
    return best


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------
def _load_cache_arrays(cache_dir: Path, rows, modalities):
    data = {}
    inchikeys = [r["inchikey"] for r in rows]
    if "smiles" in modalities:
        states = torch.load(cache_dir / "smiles_states.pt", weights_only=False)
        if states["inchikeys"] != inchikeys:
            raise ValueError("SMILES 缓存顺序与数据不一致")
        data["smiles_states"] = states["states"]
    if "confseq" in modalities:
        states = torch.load(cache_dir / "confseq_states.pt", weights_only=False)
        if states["inchikeys"] != inchikeys:
            raise ValueError("ConfSeq 缓存顺序与数据不一致")
        data["confseq_states"] = states["states"]
    return data


def _numeric_matrix(kind: str, cache_dir: Path, rows) -> np.ndarray:
    import pandas as pd
    from molpredict.padel import PADEL_NAMES
    if kind == "padel24":
        df = pd.read_csv(cache_dir / "padel24.csv")
        if len(df) != len(rows):
            raise ValueError("PaDEL 缓存行数与数据不一致")
        return df[PADEL_NAMES].to_numpy(dtype=np.float32)
    if kind == "rdkit217":
        return np.load(cache_dir / "rdkit217.npy").astype(np.float32)
    raise ValueError(f"未知数值特征: {kind}")


def _standardize_numeric(X: np.ndarray, train_idx: np.ndarray):
    train = X[train_idx].astype(np.float64)
    median = np.nanmedian(train, axis=0)
    median[~np.isfinite(median)] = 0.0
    filled = np.where(np.isfinite(X), X, median)
    mean = filled[train_idx].mean(axis=0)
    std = filled[train_idx].std(axis=0)
    std[std < 1e-6] = 1.0
    return ((filled - mean) / std).astype(np.float32), {"median": median.tolist(),
                                                        "mean": mean.tolist(),
                                                        "std": std.tolist()}


def run_neural(exp_name: str, modalities, numeric_kind, split, cache_dir: Path,
               out_dir: Path, fusion: str, seed: int, *, live_smiles=False,
               unfreeze_layers=2, epochs=60, patience=12, dim=128,
               lr=1e-3, batch_size=32, dropout=0.2,
               csv_path: Path | None = None) -> dict:
    from molpredict.modeling import load_dataset

    csv_path = csv_path or (ROOT / "data/training/confseq.csv")
    mols, labels, rows = load_dataset(csv_path)
    _seed_everything(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    numeric = None
    numeric_meta = None
    if "numeric" in modalities:
        order_path = cache_dir / "padel24_order.json"
        if order_path.exists():
            order = json.loads(order_path.read_text(encoding="utf-8"))
            if order.get("inchikeys") != [r["inchikey"] for r in rows]:
                raise ValueError(f"{numeric_kind} 缓存顺序与训练数据不一致")
        numeric = _numeric_matrix(numeric_kind, cache_dir, rows)
        numeric, numeric_meta = _standardize_numeric(numeric, split["train"])

    parts = _load_cache_arrays(cache_dir, rows, modalities)
    smiles_ids = None
    if live_smiles:
        tokens = torch.load(cache_dir / "smiles_tokens.pt", weights_only=False)
        if tokens["inchikeys"] != [r["inchikey"] for r in rows]:
            raise ValueError("SMILES token 缓存顺序与数据不一致")
        smiles_ids = tokens["input_ids"]

    def make_ds(indices):
        return V2Dataset(
            indices, labels,
            numeric=numeric,
            smiles_states=None if live_smiles else parts.get("smiles_states"),
            confseq_states=parts.get("confseq_states"),
            smiles_ids=smiles_ids,
        )

    train_ds, val_ds, test_ds = (make_ds(split[k]) for k in ("train", "val", "test"))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_v2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_v2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_v2)

    numeric_dim = numeric.shape[1] if numeric is not None else 0
    model = V2Net(
        modalities, numeric_dim, dim=dim, fusion=fusion, dropout=dropout,
        live_smiles=live_smiles, unfreeze_layers=unfreeze_layers,
    )
    pos = float(labels[split["train"]].sum())
    neg = float(len(split["train"]) - pos)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)
    )

    params = [p for p in model.parameters() if p.requires_grad]
    encoder_params = []
    if live_smiles:
        # 编码器可训练参数使用较小学习率
        encoder_params = [p for p in model.smiles_branch.encoder.parameters()
                          if p.requires_grad]
        head_names = {id(p) for p in encoder_params}
        head_params = [p for p in params if id(p) not in head_names]
        optimizer = torch.optim.AdamW([
            {"params": encoder_params, "lr": 2e-5},
            {"params": head_params, "lr": lr},
        ], weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

    best_state, best_score, best_epoch, bad = None, -1.0, -1, 0
    history = []
    started = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, n_seen = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits, batch["label"])
            loss.backward()
            nn.utils.clip_grad_norm_(params, 5.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(batch["label"])
            n_seen += len(batch["label"])
        train_loss = total_loss / max(n_seen, 1)

        model.eval()
        val_probs, val_true, val_loss = [], [], 0.0
        with torch.no_grad():
            for batch in val_loader:
                logits = model(batch)
                val_loss += float(criterion(logits, batch["label"]).item()) * len(batch["label"])
                val_probs.extend(torch.sigmoid(logits).numpy().tolist())
                val_true.extend(batch["label"].numpy().tolist())
        val_probs = np.asarray(val_probs)
        val_true = np.asarray(val_true).astype(int)
        val_auc = roc_auc_score(val_true, val_probs) if len(np.unique(val_true)) == 2 else 0.0
        score = val_auc if np.isfinite(val_auc) else -val_loss
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss / max(len(val_true), 1),
                        "val_roc_auc": float(val_auc),
                        "val_pr_auc": float(average_precision_score(val_true, val_probs))})
        if live_smiles:
            print(f"  [ft] epoch {epoch}: train_loss={train_loss:.4f} "
                  f"val_auc={val_auc:.4f}", flush=True)
        if score > best_score:
            best_score, best_epoch, bad = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    val_probs, val_true, test_probs, test_true = [], [], [], []
    with torch.no_grad():
        for batch in val_loader:
            val_probs.extend(torch.sigmoid(model(batch)).numpy().tolist())
            val_true.extend(batch["label"].numpy().tolist())
        for batch in test_loader:
            test_probs.extend(torch.sigmoid(model(batch)).numpy().tolist())
            test_true.extend(batch["label"].numpy().tolist())
    val_true = np.asarray(val_true).astype(int)
    test_true = np.asarray(test_true).astype(int)
    threshold = best_threshold(val_true, np.asarray(val_probs))
    val_metrics = binary_metrics(val_true, val_probs, threshold)
    test_metrics = binary_metrics(test_true, test_probs, threshold)

    torch.save(best_state, out_dir / "model.pt")
    _save_json(out_dir / "config.json", {
        "experiment": exp_name, "modalities": modalities, "numeric": numeric_kind,
        "fusion": fusion, "seed": seed, "live_smiles": live_smiles,
        "unfreeze_layers": unfreeze_layers, "dim": dim, "epochs": epochs,
        "patience": patience, "lr": lr, "batch_size": batch_size, "dropout": dropout,
        "best_epoch": best_epoch, "train_n": len(split["train"]),
        "val_n": len(split["val"]), "test_n": len(split["test"]),
        "train_pos": int(labels[split["train"]].sum()),
        "val_pos": int(labels[split["val"]].sum()),
        "test_pos": int(labels[split["test"]].sum()),
        "pretrained": {
            "smiles": SMILES_MODEL if "smiles" in modalities else None,
            "confseq": str(CONFSEQ_CHECKPOINT) if "confseq" in modalities else None,
        },
        "numeric_standardization": "train split only" if numeric is not None else None,
        "numeric_meta": numeric_meta,
        "elapsed_sec": round(time.time() - started, 1),
    })
    _save_json(out_dir / "metrics.json", {"val": val_metrics, "test": test_metrics,
                                          "history": history})
    with (out_dir / "predictions_test.csv").open("w", encoding="utf-8-sig",
                                                 newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "smiles", "label", "probability", "threshold", "predicted"])
        for idx, p in zip(split["test"], test_probs):
            w.writerow([rows[idx]["name"], rows[idx]["smiles"], int(labels[idx]),
                        f"{p:.6f}", f"{threshold:.6f}", int(p >= threshold)])
    print(f"[{exp_name}/{fusion}/seed{seed}] test AUC={test_metrics['roc_auc']:.4f} "
          f"PR-AUC={test_metrics['pr_auc']:.4f} F1={test_metrics['f1']:.4f} "
          f"MCC={test_metrics['mcc']:.4f} (epoch {best_epoch}, {time.time()-started:.0f}s)")
    return {"val": val_metrics, "test": test_metrics, "history": history,
            "best_epoch": best_epoch, "elapsed_sec": round(time.time() - started, 1)}


# ---------------------------------------------------------------------------
# CatBoost 基线 A（同一划分）
# ---------------------------------------------------------------------------
def run_catboost(split, out_dir: Path, seed: int, csv_path: Path | None = None) -> dict:
    from catboost import CatBoostClassifier
    from molpredict.modeling import load_dataset

    csv_path = csv_path or (ROOT / "data/training/confseq.csv")
    mols, labels, rows = load_dataset(csv_path)
    X = np.vstack([
        np.concatenate([morgan_features(m), descriptor_features(m)]) for m in mols
    ]).astype(np.float32)
    train_idx, val_idx, test_idx = split["train"], split["val"], split["test"]
    model = CatBoostClassifier(
        iterations=1000, learning_rate=0.03, depth=6, loss_function="Logloss",
        eval_metric="AUC", auto_class_weights="Balanced", random_seed=seed,
        early_stopping_rounds=100, verbose=100, thread_count=-1,
    )
    model.fit(X[train_idx], labels[train_idx],
              eval_set=(X[val_idx], labels[val_idx]), use_best_model=True)
    val_probs = model.predict_proba(X[val_idx])[:, 1]
    test_probs = model.predict_proba(X[test_idx])[:, 1]
    threshold = best_threshold(labels[val_idx], val_probs)
    metrics = {
        "val": binary_metrics(labels[val_idx], val_probs, threshold),
        "test": binary_metrics(labels[test_idx], test_probs, threshold),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out_dir / "model.cbm"))
    _save_json(out_dir / "config.json", {
        "experiment": "A", "features": "Morgan + RDKit 217", "seed": seed,
        "train_n": len(train_idx), "val_n": len(val_idx), "test_n": len(test_idx),
        "train_pos": int(labels[train_idx].sum()),
        "val_pos": int(labels[val_idx].sum()),
        "test_pos": int(labels[test_idx].sum()),
        "threshold_source": "validation set, max F1",
    })
    _save_json(out_dir / "metrics.json", metrics)
    with (out_dir / "predictions_test.csv").open("w", encoding="utf-8-sig",
                                                 newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "smiles", "label", "probability", "threshold", "predicted"])
        for idx, p in zip(test_idx, test_probs):
            w.writerow([rows[idx]["name"], rows[idx]["smiles"], int(labels[idx]),
                        f"{p:.6f}", f"{threshold:.6f}", int(p >= threshold)])
    print(f"[A/catboost/seed{seed}] test AUC={metrics['test']['roc_auc']:.4f} "
          f"PR-AUC={metrics['test']['pr_auc']:.4f} F1={metrics['test']['f1']:.4f} "
          f"MCC={metrics['test']['mcc']:.4f}")
    return metrics
