"""APPf Table S2 的 24 个 PaDEL 描述符。

数值由官方 PaDEL-Descriptor 计算（padelpy 0.1.17 内置 jar），
不使用 RDKit 同名/相似特征替代。24 个名称逐项校验。
"""

import json
from pathlib import Path

PADEL_24 = [
    ("nBr", "Number of bromine atoms"),
    ("AATS7m", "Average Broto-Moreau autocorrelation - lag 7 / weighted by mass"),
    ("ATSC0c", "Centered Broto-Moreau autocorrelation - lag 0 / weighted by charges"),
    ("AATSC5m", "Average centered Broto-Moreau autocorrelation - lag 5 / weighted by mass"),
    ("AATSC6m", "Average centered Broto-Moreau autocorrelation - lag 6 / weighted by mass"),
    ("VR1_DzZ", "Randic-like eigenvector-based index from Barysz matrix / weighted by atomic number"),
    ("nHdNH", "Count of atom-type H E-State: =NH"),
    ("nHsSH", "Count of atom-type H E-State: -SH"),
    ("nHsNH2", "Count of atom-type H E-State: -NH2"),
    ("nHsssNHp", "Count of atom-type H E-State: >NH-+"),
    ("nHCsatu", "Count of atom-type H E-State: H on C sp3 bonded to unsaturated C"),
    ("nssssNp", "Count of atom-type E-State: >N<+"),
    ("naaO", "Count of atom-type E-State: :O:"),
    ("nssssSi", "Count of atom-type E-State: >Si<"),
    ("nddssS", "Count of atom-type E-State: >S=="),
    ("SsBr", "Sum of atom-type E-State: -Br"),
    ("minHBint8", "Minimum E-State descriptors of strength for potential Hydrogen Bonds of path length 8"),
    ("minHBint9", "Minimum E-State descriptors of strength for potential Hydrogen Bonds of path length 9"),
    ("n7Ring", "Number of 7-membered rings"),
    ("nG12Ring", "Number of >12-membered rings"),
    ("nF6Ring", "Number of 6-membered fused rings"),
    ("nF7Ring", "Number of 7-membered fused rings"),
    ("nF8Ring", "Number of 8-membered fused rings"),
    ("n6HeteroRing", "Number of 6-membered rings containing heteroatoms (N, O, P, S, or halogens)"),
]
PADEL_NAMES = [name for name, _ in PADEL_24]

# 超大/聚合分子（如 Chitosan hydrochloride）会让 PaDEL 的 2D 路径/自相关
# 计算长时间卡住；这些分子保留 NaN，后续只用训练集中位数填补。
MAX_ATOMS = 60


def compute_padel24(smiles_list: list[str], cache_dir: Path) -> Path:
    """计算并缓存 24 个描述符；返回缓存 CSV 路径。"""
    import numpy as np
    import pandas as pd
    from rdkit import Chem

    cache_dir.mkdir(parents=True, exist_ok=True)
    out_csv = cache_dir / "padel24.csv"
    meta_path = cache_dir / "padel24_meta.json"
    if out_csv.exists():
        df = pd.read_csv(out_csv)
        missing = set(PADEL_NAMES) - set(df.columns)
        if missing or len(df) != len(smiles_list):
            raise ValueError(f"PaDEL 缓存不完整: missing={missing}, rows={len(df)}")
        return out_csv

    # 只计算 2D 描述符。padelpy.from_smiles 会同时计算昂贵的 3D 描述符，
    # 这里直接调用官方 CLI；这 24 个描述符全部是 2D。
    from padelpy import padeldescriptor

    keep_idx, keep_smiles, skipped = [], [], []
    for idx, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms() > MAX_ATOMS:
            skipped.append({
                "order": idx,
                "smiles": smiles,
                "n_atoms": int(mol.GetNumAtoms()) if mol is not None else None,
                "reason": f"atom count > {MAX_ATOMS} or invalid SMILES",
            })
        else:
            keep_idx.append(idx)
            keep_smiles.append(smiles)

    # 只启用包含这 24 个描述符的 PaDEL 组，避免计算 1400+ 无关描述符
    import padelpy
    import xml.etree.ElementTree as ET
    groups_xml = Path(padelpy.__file__).parent / "PaDEL-Descriptor" / "descriptors.xml"
    tree = ET.parse(groups_xml)
    keep_groups = {
        "AtomCount", "Autocorrelation", "BaryszMatrix",
        "EStateAtomType", "RingCount", "HBondAcceptorCount", "HBondDonorCount",
    }
    for group in tree.getroot().findall("Group"):
        if group.get("name") != "2D":
            continue
        for descriptor in group.findall("Descriptor"):
            descriptor.set(
                "value", "true" if descriptor.get("name") in keep_groups else "false"
            )
    groups_path = cache_dir / "padel_groups.xml"
    tree.write(groups_path, encoding="utf-8", xml_declaration=True)

    matrix = np.full((len(smiles_list), len(PADEL_NAMES)), np.nan, dtype=float)
    chunk_dir = cache_dir / "padel_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    def run_chunk(items):
        if not items:
            return
        tag = f"{items[0][0]:05d}_{items[-1][0]:05d}"
        smi_path = chunk_dir / f"{tag}.smi"
        csv_path = chunk_dir / f"{tag}.csv"
        smi_path.write_text("\n".join(s for _, s in items), encoding="utf-8")
        try:
            padeldescriptor(
                mol_dir=str(smi_path),
                d_file=str(csv_path),
                d_2d=True,
                d_3d=False,
                convert3d=False,
                threads=2,
                retainorder=True,
                descriptortypes=str(groups_path),
                sp_timeout=45,
            )
            df = pd.read_csv(csv_path, usecols=["Name"] + PADEL_NAMES)
            if len(df) != len(items):
                raise RuntimeError(f"返回 {len(df)} 行，期望 {len(items)} 行")
            values = df[PADEL_NAMES].apply(
                pd.to_numeric, errors="coerce"
            ).to_numpy(float)
            # PaDEL 某些批次会把整组 E-State 列算成空值；拆分后重算
            if len(items) > 1 and np.isnan(values).all(axis=0).any():
                raise RuntimeError("部分描述符整列为空，需要拆分批次")
            for j, (original_idx, _) in enumerate(items):
                matrix[original_idx] = values[j]
        except Exception as exc:
            if len(items) == 1:
                idx, smiles = items[0]
                skipped.append({
                    "order": idx, "smiles": smiles,
                    "n_atoms": Chem.MolFromSmiles(smiles).GetNumAtoms(),
                    "reason": f"PaDEL timeout/error: {exc}",
                })
                return
            mid = len(items) // 2
            run_chunk(items[:mid])
            run_chunk(items[mid:])

    all_items = list(zip(keep_idx, keep_smiles))
    for start in range(0, len(all_items), 20):
        run_chunk(all_items[start:start + 20])

    out = pd.DataFrame({"order": range(len(smiles_list)), "smiles": smiles_list})
    for j, name in enumerate(PADEL_NAMES):
        out[name] = matrix[:, j]
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    meta_path.write_text(json.dumps({
        "source": "PaDEL-Descriptor via padelpy 0.1.17 (PaDEL-Descriptor.jar)",
        "reference": "Yap CW. PaDEL-Descriptor. J Comput Chem. 2011;32:1466-1474.",
        "license": "PaDEL-Descriptor jar: AGPL-3.0 (see padelpy/PaDEL-Descriptor/license)",
        "descriptors": [{"name": n, "description": d} for n, d in PADEL_24],
        "rows": len(out),
        "max_atoms": MAX_ATOMS,
        "skipped_molecules": skipped,
        "nan_counts": {n: int(out[n].isna().sum()) for n in PADEL_NAMES},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_csv
