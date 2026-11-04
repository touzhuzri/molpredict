"""Shared paths, structure normalization and metric helpers.

All outputs stay under experiments/merged_fungicide.  The old project and
pesticide-data are treated as read-only inputs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments" / "merged_fungicide"
DATA = OUT / "data"
FEATURES = OUT / "features"
SPLITS = OUT / "splits"
CHECKPOINTS = OUT / "checkpoints"
METRICS = OUT / "metrics"
LOGS = OUT / "logs"

OLD_LABELED = ROOT / "data" / "training" / "ppdb_labeled.csv"
OLD_CONFSEQ = ROOT / "data" / "training" / "confseq.csv"
NEW_ROOT = ROOT / "shujv" / "pesticide-data" / "data" / "processed" / "d2_1_training"
NEW_MOLECULES = NEW_ROOT / "molecules.csv"
NEW_LABELS = NEW_ROOT / "labels.csv"
NEW_EVIDENCE = NEW_ROOT / "label_evidence.csv"
NEW_ELIGIBLE = NEW_ROOT / "training_eligible.csv"

FUNGICIDE = "FUNGICIDE"
TARGET_USES = [FUNGICIDE, "INSECTICIDE", "HERBICIDE", "ACARICIDE",
               "NEMATICIDE", "RODENTICIDE", "MOLLUSCICIDE", "BACTERICIDE",
               "PLANT_GROWTH_REGULATOR", "FUMIGANT"]

# Scaffold groups with at least this many structures are forced into train.
# A single benzene-like Murcko scaffold otherwise takes over the test split.
FORCE_TRAIN_MIN_GROUP = 20
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dump_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                              allow_nan=False),
                    encoding="utf-8")


def canonicalize(smiles: str | None) -> str | None:
    """Canonical isomeric SMILES, or None if RDKit cannot parse it."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def scaffold_key(smiles: str | None) -> str | None:
    """Murcko scaffold of the largest fragment.

    Acyclic compounds have no ring scaffold; each exact acyclic structure is
    its own group.  This keeps all identical structures together without
    putting every acyclic molecule into one giant group.
    """
    if not isinstance(smiles, str) or not smiles:
        return None
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True)
    main = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    try:
        scaf = MurckoScaffold.MurckoScaffoldSmiles(
            mol=main, includeChirality=False
        )
    except Exception:
        scaf = ""
    if not scaf:
        scaf = "ACYCLIC:" + Chem.MolToSmiles(main, isomericSmiles=False)
    return scaf


def binary_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    from sklearn.metrics import (
        average_precision_score, confusion_matrix, f1_score,
        matthews_corrcoef, precision_score, recall_score, roc_auc_score,
    )
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba, dtype=float)
    pred = (p >= threshold).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    out = {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "negatives": int((1 - y).sum()),
        "threshold": float(threshold),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "confusion_matrix": {
            "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
        },
    }
    if len(np.unique(y)) == 2:
        out["roc_auc"] = float(roc_auc_score(y, p))
        out["pr_auc"] = float(average_precision_score(y, p))
    else:
        out["roc_auc"] = None
        out["pr_auc"] = None
    return out


def best_threshold(y_true, proba) -> float:
    from sklearn.metrics import f1_score
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba, dtype=float)
    thresholds = np.unique(np.round(p, 4))
    best, best_f1 = 0.5, -1.0
    for t in thresholds:
        score = f1_score(y, (p >= t).astype(int), zero_division=0)
        if score > best_f1:
            best, best_f1 = float(t), score
    return best
