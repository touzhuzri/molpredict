"""构建新数据辅助训练集（只读新项目；输出到 new_data_comparison）。

筛选规则：
- training_eligible + is_parseable + SINGLE_COMPONENT；
- 排除与旧项目任何分子相同的 InChIKey；
- 按 InChIKey 去重；
- 标记 MAIN/EXTENDED；
- strict = Murcko scaffold 不在旧 val/test 的任何 scaffold 中；
- 计算 ChemBERTa token states 和 24 项 PaDEL；NaN 保留给辅助 mask。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from molpredict.padel import PADEL_NAMES, compute_padel24  # noqa: E402
from molpredict.v3_data import load_rows, load_split  # noqa: E402
from molpredict.v3_encoders import (  # noqa: E402
    encode_smiles_states,
    load_smiles_encoder,
    load_smiles_tokenizer,
)

NEW = ROOT / "shujv" / "pesticide-data" / "data" / "processed" / "d2_1_training"
OUT = ROOT / "experiments" / "new_data_comparison"


def _scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)


def main() -> int:
    out_data = OUT / "data_aux"
    out_data.mkdir(parents=True, exist_ok=True)
    audit_dir = OUT / "data_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    _, _, old_rows = load_rows()
    old_keys = {r["inchikey"] for r in old_rows}
    old_scaffolds = {_scaffold(r["smiles"]) for r in old_rows}
    valtest_scaffolds = set()
    for seed in (42, 43, 44):
        split = load_split(seed)
        for idx in list(split["val"]) + list(split["test"]):
            valtest_scaffolds.add(_scaffold(old_rows[int(idx)]["smiles"]))

    molecules = pd.read_csv(
        NEW / "molecules.csv", encoding="utf-8-sig",
        usecols=["substance_id", "structure_form"],
    )
    eligible = pd.read_csv(NEW / "training_eligible.csv", encoding="utf-8-sig")
    df = eligible.merge(molecules, on="substance_id", how="left")
    audit = {
        "source": str(NEW / "training_eligible.csv"),
        "eligible_rows": int(len(df)),
        "tier_counts_raw": df["eligibility_tier"].value_counts().to_dict(),
        "structure_status_raw": df["structure_status"].value_counts().to_dict(),
        "identity_status_raw": df["identity_status"].value_counts().to_dict(),
        "structure_form_raw": df["structure_form"].value_counts().to_dict(),
        "unique_inchikey_raw": int(df["inchikey"].nunique()),
        "duplicate_rows_raw": int(df["inchikey"].duplicated().sum()),
        "old_overlap_rows": int(df["inchikey"].isin(old_keys).sum()),
        "old_overlap_unique": int(df.loc[df["inchikey"].isin(old_keys),
                                        "inchikey"].nunique()),
    }

    df = df[(df["is_parseable"] == 1)
            & (df["structure_form"] == "SINGLE_COMPONENT")]
    audit["after_single_component"] = int(len(df))
    df = df[~df["inchikey"].isin(old_keys)]
    audit["after_excluding_old"] = int(len(df))
    df = df.drop_duplicates("inchikey").reset_index(drop=True)
    audit["after_dedup"] = int(len(df))
    audit["tier_counts_dedup"] = df["eligibility_tier"].value_counts().to_dict()

    df["scaffold"] = df["smiles"].map(_scaffold)
    df["strict"] = ~df["scaffold"].isin(valtest_scaffolds)
    audit["scaffold_overlap_old_valtest"] = int((~df["strict"]).sum())
    audit["strict_counts"] = (
        df[df["strict"]]["eligibility_tier"].value_counts().to_dict()
    )
    audit["exact_counts"] = df["eligibility_tier"].value_counts().to_dict()
    df.to_csv(audit_dir / "new_aux_candidates.csv", index=False,
              encoding="utf-8-sig")

    # ChemBERTa token states（冻结）
    tokenizer = load_smiles_tokenizer()
    encoder = load_smiles_encoder(0)
    states, ids, masks, truncated = encode_smiles_states(
        df["smiles"].tolist(), tokenizer, encoder, max_length=320,
    )
    audit["smiles_truncated"] = int(truncated)
    audit["smiles_length_max"] = int(max(len(x) for x in ids))
    audit["smiles_length_mean"] = float(np.mean([len(x) for x in ids]))

    # 24 PaDEL
    padel_csv = compute_padel24(
        df["smiles"].tolist(), out_data / "padel_cache"
    )
    padel = pd.read_csv(padel_csv, encoding="utf-8-sig")
    if len(padel) != len(df):
        raise RuntimeError("PaDEL 行数与候选不一致")
    if not (padel["smiles"].astype(str).tolist()
            == df["smiles"].astype(str).tolist()):
        raise RuntimeError("PaDEL 顺序与候选不一致")
    descriptors = padel[PADEL_NAMES].to_numpy(dtype=np.float32)
    audit["padel_nan_counts"] = {
        name: int(np.isnan(descriptors[:, i]).sum())
        for i, name in enumerate(PADEL_NAMES)
    }
    audit["padel_rows_with_any_nan"] = int(
        np.isnan(descriptors).any(axis=1).sum()
    )

    torch.save({
        "substance_id": df["substance_id"].tolist(),
        "name": df["canonical_name"].tolist(),
        "smiles": df["smiles"].tolist(),
        "inchikey": df["inchikey"].tolist(),
        "tier": df["eligibility_tier"].tolist(),
        "strict": df["strict"].astype(bool).tolist(),
        "scaffold": df["scaffold"].tolist(),
        "states": states,
        "input_ids": ids,
        "attention_mask": masks,
        "descriptors": descriptors,
        "padel_names": PADEL_NAMES,
    }, out_data / "new_aux.pt")
    (audit_dir / "new_aux_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
