"""Feature builders for the APPf-like CatBoost experiments.

The faithful APPf replication uses exactly the 24 PaDEL descriptors of the
original project.  The extended demonstration model adds 12 cheap RDKit
descriptors plus Morgan and MACCS fingerprints; it is still a CatBoost model
and is reported separately so the replication is not confused with the
extended model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import DATA, FEATURES, SPLITS, canonicalize  # noqa: E402
from molpredict.padel import PADEL_NAMES  # noqa: E402

RDKIT_NAMES = [
    "MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds", "RingCount", "NumAromaticRings",
    "FractionCSP3", "NumHeteroatoms", "NumBridgeheadAtoms",
    "NumSpiroAtoms",
]
MORGAN_N_BITS = 2048
MORGAN_RADIUS = 2
MACCS_N_BITS = 167
FEATURE_SETS = {
    "P24": {"padel": True, "rdkit": False, "morgan": False, "maccs": False},
    "EXT": {"padel": True, "rdkit": True, "morgan": True, "maccs": True},
}


def rdkit_descriptors(mol) -> np.ndarray:
    from rdkit.Chem import Descriptors, rdMolDescriptors

    return np.asarray([
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.RingCount(mol),
        Descriptors.NumAromaticRings(mol),
        Descriptors.FractionCSP3(mol),
        rdMolDescriptors.CalcNumHeteroatoms(mol),
        rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
        rdMolDescriptors.CalcNumSpiroAtoms(mol),
    ], dtype=np.float32)


def morgan_fingerprint(mol) -> np.ndarray:
    from rdkit.Chem import AllChem

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, MORGAN_RADIUS, nBits=MORGAN_N_BITS
    )
    return np.frombuffer(fp.ToBitString().encode(), dtype=np.uint8) - ord("0")


def maccs_fingerprint(mol) -> np.ndarray:
    from rdkit.Chem import MACCSkeys

    fp = MACCSkeys.GenMACCSKeys(mol)
    return np.frombuffer(fp.ToBitString().encode(), dtype=np.uint8) - ord("0")


def feature_names(feature_set: str) -> list[str]:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"unknown feature set: {feature_set}")
    cfg = FEATURE_SETS[feature_set]
    names = []
    if cfg["padel"]:
        names += [f"PaDEL:{n}" for n in PADEL_NAMES]
    if cfg["rdkit"]:
        names += [f"RDKit:{n}" for n in RDKIT_NAMES]
    if cfg["morgan"]:
        names += [f"Morgan:{i}" for i in range(MORGAN_N_BITS)]
    if cfg["maccs"]:
        names += [f"MACCS:{i}" for i in range(MACCS_N_BITS)]
    return names


def build_structure_arrays(structures: pd.DataFrame) -> dict:
    """Return per-structure feature arrays in structures.csv row order."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    padel = np.load(FEATURES / "padel24.npy").astype(np.float32)
    if padel.shape != (len(structures), len(PADEL_NAMES)):
        raise ValueError(f"PaDEL shape mismatch: {padel.shape}")
    mols = [Chem.MolFromSmiles(s) for s in structures["canonical_smiles"]]
    if any(m is None for m in mols):
        raise ValueError("structures.csv contains unparseable canonical SMILES")
    rdkit = np.vstack([rdkit_descriptors(m) for m in mols]).astype(np.float32)
    morgan = np.vstack([morgan_fingerprint(m) for m in mols]).astype(np.uint8)
    maccs = np.vstack([maccs_fingerprint(m) for m in mols]).astype(np.uint8)
    return {
        "padel": padel, "rdkit": rdkit,
        "morgan": morgan, "maccs": maccs,
    }


def model_matrix(table: pd.DataFrame, arrays: dict,
                 feature_set: str) -> np.ndarray:
    cfg = FEATURE_SETS[feature_set]
    idx = table["structure_index"].to_numpy(dtype=int)
    parts = []
    if cfg["padel"]:
        parts.append(arrays["padel"][idx].astype(np.float32))
    if cfg["rdkit"]:
        parts.append(arrays["rdkit"][idx].astype(np.float32))
    if cfg["morgan"]:
        parts.append(arrays["morgan"][idx].astype(np.float32))
    if cfg["maccs"]:
        parts.append(arrays["maccs"][idx].astype(np.float32))
    X = np.concatenate(parts, axis=1).astype(np.float32)
    names = feature_names(feature_set)
    if X.shape[1] != len(names):
        raise ValueError(f"{feature_set}: X has {X.shape[1]} columns, "
                         f"{len(names)} names")
    return X


def load_split(split_seed: int, table: pd.DataFrame) -> dict:
    split = json.loads(
        (SPLITS / f"split_seed{split_seed}.json").read_text(encoding="utf-8")
    )
    out = {k: np.asarray(split[k], dtype=int) for k in ("train", "val", "test")}
    # Validate that the JSON indices and the modeling table agree.
    seen = np.concatenate([out["train"], out["val"], out["test"]])
    if len(seen) != len(table) or set(seen.tolist()) != set(range(len(table))):
        raise ValueError("split indices do not partition the modeling table")
    return out


def balanced_bags(train_idx: np.ndarray, labels: np.ndarray,
                  n_bags: int = 25, seed: int = 1000) -> list[np.ndarray]:
    """One balanced subsample per bag: all positives + equal negatives."""
    train_idx = np.asarray(train_idx, dtype=int)
    y = np.asarray(labels, dtype=int)
    pos = train_idx[y[train_idx] == 1]
    neg = train_idx[y[train_idx] == 0]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("balanced sampling requires both classes")
    n = min(len(pos), len(neg))
    bags = []
    for b in range(n_bags):
        rng = np.random.default_rng(seed + b)
        p = rng.choice(pos, size=n, replace=False)
        q = rng.choice(neg, size=n, replace=False)
        bags.append(np.sort(np.concatenate([p, q])))
    return bags


def tanimoto_to_train(query_morgan: np.ndarray,
                      train_morgan: np.ndarray) -> np.ndarray:
    """Maximum Tanimoto similarity of each query to any training structure."""
    from rdkit import DataStructs

    def to_fp(row):
        fp = DataStructs.ExplicitBitVect(MORGAN_N_BITS)
        fp.SetBitsFromList(np.flatnonzero(row).tolist())
        return fp

    train_fps = [to_fp(r) for r in train_morgan]
    return np.asarray([
        max(DataStructs.BulkTanimotoSimilarity(to_fp(q), train_fps))
        for q in query_morgan
    ], dtype=float)


def save_feature_spec(path: Path) -> None:
    payload = {
        "padel_names": PADEL_NAMES,
        "rdkit_names": RDKIT_NAMES,
        "morgan": {"radius": MORGAN_RADIUS, "n_bits": MORGAN_N_BITS},
        "maccs": {"n_bits": MACCS_N_BITS},
        "feature_sets": {
            k: {"n_features": len(feature_names(k)), "columns": feature_names(k)}
            for k in FEATURE_SETS
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8")
