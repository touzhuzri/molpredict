"""M1：批量读取 SMILES，生成 InChIKey、2D PNG 和 3D SDF。"""

import csv
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem, Draw

CSV_COLUMNS = ["record_id", "name", "smiles", "inchikey", "image_2d"]
FAILED_COLUMNS = ["name", "smiles", "reason"]


def _read_rows(csv_path: Path) -> list[dict]:
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"{csv_path} 为空")
        missing = {"name", "smiles"} - {c.strip().lower() for c in reader.fieldnames}
        if missing:
            raise ValueError(f"CSV 缺少列: {', '.join(sorted(missing))}")
        reader.fieldnames = [c.strip().lower() for c in reader.fieldnames]
        return [row for row in reader if any((v or "").strip() for v in row.values())]


def _make_3d(mol: Chem.Mol) -> Chem.Mol:
    """用 RDKit 生成并优化一个 3D 构象，返回带氢原子的分子。"""
    mol3d = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(mol3d, params) != 0:
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol3d, params) != 0:
            raise RuntimeError("3D 构象生成失败")
    if AllChem.MMFFOptimizeMolecule(mol3d) == -1:
        AllChem.UFFOptimizeMolecule(mol3d)
    return mol3d


def process_csv(csv_path: Path, out_dir: Path) -> dict:
    """处理 CSV，写出成功/失败记录以及 2D/3D 结构。"""
    rows = _read_rows(csv_path)

    structures_2d = out_dir / "structures_2d"
    structures_3d = out_dir / "structures_3d"
    structures_2d.mkdir(parents=True, exist_ok=True)
    structures_3d.mkdir(parents=True, exist_ok=True)
    sdf_path = structures_3d / "molecules.sdf"

    successes, failures = [], []
    seen: dict[str, str] = {}

    writer = Chem.SDWriter(str(sdf_path))
    try:
        for i, row in enumerate(rows, start=1):
            name = (row.get("name") or "").strip() or f"row_{i}"
            smiles = (row.get("smiles") or "").strip()
            record_id = f"molecule_{i:03d}"
            try:
                mol = Chem.MolFromSmiles(smiles) if smiles else None
                if mol is None:
                    raise ValueError("SMILES 无法解析")

                inchikey = Chem.MolToInchiKey(mol)
                if not inchikey:
                    raise ValueError("InChIKey 生成失败")
                if inchikey in seen:
                    raise ValueError(f"重复分子（与 {seen[inchikey]} 的 InChIKey 相同）")
                seen[inchikey] = name

                mol2d = Chem.Mol(mol)
                AllChem.Compute2DCoords(mol2d)
                image_rel = f"structures_2d/{record_id}.png"
                Draw.MolToFile(mol2d, str(out_dir / image_rel), size=(400, 300))

                mol3d = _make_3d(mol)
                mol3d.SetProp("_Name", name)
                mol3d.SetProp("RECORD_ID", record_id)
                mol3d.SetProp("SMILES", smiles)
                mol3d.SetProp("INCHIKEY", inchikey)
                writer.write(mol3d)

                successes.append({
                    "record_id": record_id,
                    "name": name,
                    "smiles": smiles,
                    "inchikey": inchikey,
                    "image_2d": image_rel,
                })
                print(f"[OK]   {name}  {inchikey}")
            except Exception as exc:
                failures.append({"name": name, "smiles": smiles, "reason": str(exc)})
                print(f"[FAIL] {name}: {exc}")
    finally:
        writer.close()

    with (out_dir / "molecules.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(successes)

    with (out_dir / "failed.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FAILED_COLUMNS)
        w.writeheader()
        w.writerows(failures)

    return {
        "total": len(rows),
        "success": len(successes),
        "failed": len(failures),
        "out_dir": out_dir,
        "sdf_path": sdf_path,
    }
