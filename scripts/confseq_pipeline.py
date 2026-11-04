"""用 ConfSeq 官方 demo 代码把 3D 构象转换为 ConfSeq 字符串。

不实现替代编码；只复用 data/external/ConfSeq/demo/ConfSeq.py。
生成 CSV 后再调用官方 get_confseq_embeddings.py 得到 256 维 embedding。
"""

import argparse
import csv
import signal
import sys
from multiprocessing import Pool
from pathlib import Path

import timeout_decorator
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data" / "external" / "ConfSeq"))

# ConfSeq 官方 demo 用 signal.SIGALRM 做超时；Windows 没有该信号，改为不设超时
if not hasattr(signal, "SIGALRM"):
    timeout_decorator.timeout = lambda *args, **kwargs: (lambda func: func)

from demo.ConfSeq import run_aug_mol_get_ConfSeq_pair_0  # noqa: E402
from molpredict.molecules import _make_3d  # noqa: E402


def process_row(row: dict):
    smiles = (row.get("smiles") or "").strip()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return row, "", "SMILES 无法解析"
    try:
        mol3d = Chem.RemoveHs(_make_3d(mol))
        result = run_aug_mol_get_ConfSeq_pair_0(
            (mol3d, Chem.MolToSmiles(mol3d))
        )
    except Exception as exc:
        return row, "", str(exc)
    parts = (result or "").split("\t")
    if len(parts) != 3 or not parts[2].strip():
        return row, "", "ConfSeq 转换失败"
    return row, parts[2].strip(), ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "training" / "confseq.csv")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    with args.csv.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    with Pool(processes=args.workers) as pool:
        results = pool.map(process_row, rows, chunksize=8)

    records, failed = [], []
    for row, confseq, error in results:
        if error:
            failed.append({**row, "reason": error})
            continue
        records.append({**row, "confseq": confseq})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(records[0].keys()) if records else []
    with args.out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    failed_path = args.out.with_name(args.out.stem + "_failed.csv")
    with failed_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) + ["reason"])
        writer.writeheader()
        writer.writerows(failed)

    print(f"读取 {len(rows)}，成功 {len(records)}，失败 {len(failed)}")
    print(f"写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
