"""M2/M3：特征提取、CatBoost 训练与预测。"""

import csv
import json
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from catboost import CatBoostClassifier

RDLogger.DisableLog("rdApp.*")

MORGAN_BITS = 2048
MORGAN_RADIUS = 2
DESCRIPTOR_NAMES = [name for name, _ in Descriptors.descList]


def morgan_features(mol: Chem.Mol) -> np.ndarray:
    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=MORGAN_RADIUS, fpSize=MORGAN_BITS
    )
    fp = gen.GetFingerprint(mol)
    arr = np.zeros(MORGAN_BITS, dtype=np.float32)
    arr[list(fp.GetOnBits())] = 1.0
    return arr


def descriptor_features(mol: Chem.Mol) -> np.ndarray:
    values = []
    for _, func in Descriptors.descList:
        try:
            value = float(func(mol))
        except Exception:
            value = np.nan
        values.append(value if np.isfinite(value) else np.nan)
    return np.asarray(values, dtype=np.float32)


def featurize(mol: Chem.Mol, feature: str) -> np.ndarray:
    if feature == "morgan":
        return morgan_features(mol)
    if feature == "descriptors":
        return descriptor_features(mol)
    if feature == "combined":
        return np.concatenate([morgan_features(mol), descriptor_features(mol)])
    if feature in ("confseq", "all"):
        raise ValueError(f"{feature} 特征由 embedding 文件提供，不能从单个 SMILES 直接计算")
    raise ValueError(f"未知特征类型: {feature}")


def load_dataset(csv_path: Path) -> tuple[list[Chem.Mol], np.ndarray, list[dict]]:
    mols, labels, rows = [], [], []
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            smiles = (row.get("smiles") or "").strip()
            if not smiles or row.get("label", "").strip() == "":
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            mols.append(mol)
            labels.append(int(row["label"]))
            rows.append(row)
    if not mols:
        raise ValueError(f"{csv_path} 中没有可用的带标签分子")
    return mols, np.asarray(labels, dtype=int), rows


def scaffold_groups(mols: list[Chem.Mol]) -> list[str]:
    groups = []
    for mol in mols:
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        except Exception:
            scaffold = ""
        # 无环分子没有 Murcko scaffold，用自身 SMILES 保证每个分子独立成组
        groups.append(scaffold or Chem.MolToSmiles(mol))
    return groups


def make_split(mols, labels, seed=42, test_size=0.2):
    """按 Murcko scaffold 分组划分，避免同骨架分子泄漏到测试集。"""
    groups = scaffold_groups(mols)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(mols, labels, groups))
    return np.array(train_idx), np.array(test_idx)


def evaluate(y_true, proba, threshold=0.5) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
    }


def train_one(feature, mols, labels, rows, train_idx, test_idx, out_dir: Path,
              seed=42, X=None):
    if X is None:
        X = np.vstack([featurize(mol, feature) for mol in mols])
    X = np.asarray(X, dtype=np.float32)
    y = labels

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        raise ValueError(f"{feature}: 划分后某一侧只有单一类别，无法训练/评估")

    # 仅用训练集内部验证做早停，测试集只用于最终评估
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=seed, stratify=y_train
    )

    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights="Balanced",
        random_seed=seed,
        early_stopping_rounds=100,
        verbose=100,
        thread_count=-1,
    )
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)

    proba = model.predict_proba(X_test)[:, 1]
    metrics = evaluate(y_test, proba)
    metrics.update({
        "feature": feature,
        "train_n": int(len(train_idx)),
        "train_positives": int(y_train.sum()),
        "test_n": int(len(test_idx)),
        "model_path": str(out_dir / f"{feature}_catboost.cbm"),
    })

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out_dir / f"{feature}_catboost.cbm"))
    with (out_dir / f"predictions_{feature}.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["name", "smiles", "label", "probability"])
        for idx, p in zip(test_idx, proba):
            writer.writerow([
                rows[idx].get("name", ""),
                rows[idx].get("smiles", ""),
                int(y[idx]),
                f"{p:.6f}",
            ])
    print(
        f"{feature}: AUC={metrics['roc_auc']:.4f} PR-AUC={metrics['pr_auc']:.4f} "
        f"F1={metrics['f1']:.4f} P={metrics['precision']:.4f} R={metrics['recall']:.4f}"
    )
    return metrics, model


def train(csv_path: Path, out_dir: Path, features: list[str], seed=42,
          embeddings_path: Path | None = None) -> dict:
    mols, labels, rows = load_dataset(csv_path)
    train_idx, test_idx = make_split(mols, labels, seed=seed)
    print(
        f"数据: {len(mols)} 个分子，阳性 {int(labels.sum())}，阴性 {int(len(labels)-labels.sum())}"
        f"；训练 {len(train_idx)}，测试 {len(test_idx)}（Murcko scaffold 分组划分）"
    )
    results = {}
    for feature in features:
        X = None
        if feature in ("confseq", "all"):
            if embeddings_path is None:
                raise ValueError(f"{feature} 特征需要提供 --embeddings（.npy 文件）")
            embeddings = np.load(embeddings_path)
            if len(embeddings) != len(mols):
                raise ValueError(
                    f"embedding 行数 {len(embeddings)} 与分子数 {len(mols)} 不一致，"
                    "请确认 embedding CSV 与训练 CSV 行序一致"
                )
            if feature == "confseq":
                X = embeddings
            else:
                base = np.vstack([featurize(mol, "combined") for mol in mols])
                X = np.hstack([base, embeddings])
        metrics, _ = train_one(
            feature, mols, labels, rows, train_idx, test_idx, out_dir, seed, X=X
        )
        results[feature] = metrics

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"
    existing = {}
    if metrics_path.exists():
        existing = json.loads(metrics_path.read_text(encoding="utf-8"))
    existing.update(results)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    with (out_dir / "feature_meta.json").open("w", encoding="utf-8") as f:
        json.dump({
            "morgan_bits": MORGAN_BITS,
            "morgan_radius": MORGAN_RADIUS,
            "descriptor_names": DESCRIPTOR_NAMES,
            "split": "Murcko scaffold GroupShuffleSplit",
            "seed": seed,
        }, f, ensure_ascii=False, indent=2)
    return results


def predict_smiles(smiles: str, model_path: Path, feature: str, threshold=0.5) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("SMILES 无法解析")
    model = CatBoostClassifier()
    model.load_model(str(model_path))
    x = featurize(mol, feature).reshape(1, -1)
    proba = float(model.predict_proba(x)[0, 1])
    return {
        "smiles": smiles,
        "inchikey": Chem.MolToInchiKey(mol),
        "probability": proba,
        "prediction": "可能具有杀菌活性" if proba >= threshold else "可能不具有杀菌活性",
        "threshold": threshold,
        "feature": feature,
    }
