"""Score an Excel job-list export with the APPf-like EXT model.

The workbook is read-only.  Outputs are written under
experiments/merged_fungicide/appf_like/predictions/excel_runs/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from rdkit import Chem, DataStructs, RDLogger
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from appf_features import (  # noqa: E402
    MORGAN_N_BITS, maccs_fingerprint, morgan_fingerprint, rdkit_descriptors,
)
from common import DATA, OUT, canonicalize  # noqa: E402
from molpredict.padel import PADEL_NAMES, compute_padel24  # noqa: E402

RDLogger.DisableLog("rdApp.*")

APPF = OUT / "appf_like"
MODEL_DIR = APPF / "models" / "EXT" / "split42"
PRED = APPF / "predictions" / "excel_runs"
CACHE = APPF / "demo_cache" / "excel_padel"


def read_job_excel(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, header=None)
    header_idx = None
    for i in range(min(len(raw), 50)):
        values = [str(x).strip() for x in raw.iloc[i].tolist()]
        if "Job ID" in values and "SMILES" in values:
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("Could not find a header row containing Job ID and SMILES")
    header = [str(x).strip() for x in raw.iloc[header_idx].tolist()]
    df = raw.iloc[header_idx + 1:].copy()
    df.columns = header
    df = df.dropna(how="all")
    df = df.reset_index(drop=True)
    df["excel_row"] = np.arange(header_idx + 2, header_idx + 2 + len(df))
    return df


def load_models() -> list[CatBoostClassifier]:
    paths = sorted(MODEL_DIR.glob("bag*.cbm"))
    if len(paths) != 25:
        raise RuntimeError(f"expected 25 models in {MODEL_DIR}, found {len(paths)}")
    models = []
    for p in paths:
        m = CatBoostClassifier()
        m.load_model(str(p))
        models.append(m)
    return models


def load_calibrator() -> IsotonicRegression:
    val = pd.read_csv(APPF / "predictions" / "EXT_split42_val.csv")
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(val["prob_raw"].to_numpy(float), val["label"].to_numpy(int))
    return iso


def max_tanimoto(query: np.ndarray, train: np.ndarray) -> np.ndarray:
    def to_fp(row):
        fp = DataStructs.ExplicitBitVect(MORGAN_N_BITS)
        fp.SetBitsFromList(np.flatnonzero(row).tolist())
        return fp

    train_fps = [to_fp(r) for r in train]
    return np.asarray([
        max(DataStructs.BulkTanimotoSimilarity(to_fp(q), train_fps))
        for q in query
    ], dtype=float)


def padel_matrix(canonical: list[str]) -> np.ndarray:
    CACHE.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1("\n".join(canonical).encode("utf-8")).hexdigest()[:16]
    cache_dir = CACHE / key
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path = compute_padel24(canonical, cache_dir)
    df = pd.read_csv(csv_path).sort_values("order")
    vals = df[PADEL_NAMES].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if vals.shape != (len(canonical), len(PADEL_NAMES)):
        raise RuntimeError(f"PaDEL shape mismatch: {vals.shape}")
    return vals.astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("excel", type=Path)
    parser.add_argument("--sheet", default=None)
    args = parser.parse_args()

    path = args.excel.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    if args.sheet is None:
        df = read_job_excel(path)
    else:
        df = pd.read_excel(path, sheet_name=args.sheet, header=10)
    if "SMILES" not in df.columns:
        raise ValueError("Excel must contain a SMILES column")
    total_rows = len(df)

    # Canonicalize every non-empty SMILES.
    rows = []
    for i, row in df.iterrows():
        raw = row.get("SMILES")
        text = "" if pd.isna(raw) else str(raw).strip()
        if not text:
            rows.append({
                "row_index": int(i), "excel_row": row.get("excel_row", i + 1),
                "raw_smiles": None, "canonical_smiles": None,
                "valid": False, "error": "missing SMILES",
                "mol": None,
            })
            continue
        can = canonicalize(text)
        mol = Chem.MolFromSmiles(can) if can else None
        rows.append({
            "row_index": int(i), "excel_row": row.get("excel_row", i + 1),
            "raw_smiles": text, "canonical_smiles": can,
            "valid": mol is not None,
            "error": None if mol is not None else "invalid/truncated SMILES",
            "mol": mol,
        })

    valid = [r for r in rows if r["valid"]]
    unique_can = sorted({r["canonical_smiles"] for r in valid})
    print(f"rows={total_rows} valid_rows={len(valid)} "
          f"unique_valid={len(unique_can)} invalid_rows={total_rows - len(valid)}")

    # Feature matrix for unique valid structures.
    mols = [Chem.MolFromSmiles(s) for s in unique_can]
    padel = padel_matrix(unique_can)
    rd = np.vstack([rdkit_descriptors(m) for m in mols]).astype(np.float32)
    morgan = np.vstack([morgan_fingerprint(m) for m in mols]).astype(np.uint8)
    maccs = np.vstack([maccs_fingerprint(m) for m in mols]).astype(np.uint8)
    X = np.concatenate([padel, rd, morgan, maccs], axis=1).astype(np.float32)
    X[~np.isfinite(X)] = np.nan
    mol_weight = rd[:, 0].astype(float)
    heavy_atoms = np.asarray([m.GetNumHeavyAtoms() for m in mols], dtype=int)
    # Flag solvent/reagent/fragment-like entries.  These are kept in the
    # output but should not be the first thing a screening chemist looks at.
    fragment_like = (mol_weight < 150.0) | (heavy_atoms < 10)

    # Predict with the 25 balanced EXT models and calibrate on validation.
    models = load_models()
    raw = np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)
    raw_std = np.std([m.predict_proba(X)[:, 1] for m in models], axis=0)
    iso = load_calibrator()
    cal = iso.predict(raw)
    train_fp = np.load(APPF / "features" / "train_morgan_split42.npy")
    maxsim = max_tanimoto(morgan, train_fp)

    summary_json = json.loads(
        (APPF / "metrics" / "summary.json").read_text(encoding="utf-8")
    )
    threshold = float(summary_json["split42_selected"]["threshold"])
    ad_threshold = float(
        summary_json["applicability_domain"]["strict_threshold"]
    )

    unique = pd.DataFrame({
        "canonical_smiles": unique_can,
        "priority_score": cal,
        "raw_score": raw,
        "bag_sd": raw_std,
        "above_threshold": cal >= threshold,
        "in_domain": maxsim >= ad_threshold,
        "max_train_tanimoto": maxsim,
        "n_rows": 0,
        "n_jobs": 0,
        "in_merged_dataset": False,
        "merged_label_status": "",
        "merged_label": np.nan,
        "mol_weight": mol_weight,
        "heavy_atoms": heavy_atoms,
        "fragment_like": fragment_like,
    })
    # Exact-match annotations against the merged dataset.
    structures = pd.read_csv(
        DATA / "structures.csv", low_memory=False
    ).set_index("structure_id")
    structural = set(structures.index)
    for j, s in enumerate(unique_can):
        if s in structural:
            row = structures.loc[s]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            unique.at[j, "in_merged_dataset"] = True
            unique.at[j, "merged_label_status"] = str(row["label_status"])
            if pd.notna(row["label"]):
                unique.at[j, "merged_label"] = int(row["label"])
    modeling = pd.read_csv(
        DATA / "modeling_structures_seed42.csv", low_memory=False
    )
    split_map = dict(zip(modeling["structure_id"], modeling["split"]))
    unique["modeling_split"] = unique["canonical_smiles"].map(split_map).fillna("")
    unique["in_training"] = unique["modeling_split"].eq("train")
    unique["rank_raw"] = unique["raw_score"].rank(
        ascending=False, method="min"
    ).astype(int)
    unique["rank_calibrated"] = unique["priority_score"].rank(
        ascending=False, method="min"
    ).astype(int)
    # Keep rank_unique as the raw-score rank: isotonic calibration creates
    # plateaus, and raw ranking is more useful for ordering candidates.
    unique["rank_unique"] = unique["rank_raw"]
    unique = unique.sort_values("rank_raw").reset_index(drop=True)

    score_map = unique.set_index("canonical_smiles")[
        ["priority_score", "raw_score", "bag_sd", "above_threshold",
         "in_domain", "max_train_tanimoto", "rank_unique", "rank_raw",
         "rank_calibrated", "in_merged_dataset", "merged_label_status",
         "merged_label", "mol_weight", "heavy_atoms", "fragment_like",
         "modeling_split", "in_training"]
    ].to_dict("index")

    # All original rows plus predictions.
    out_rows = []
    for r, original in zip(rows, df.to_dict("records")):
        base = {
            "excel_row": r["excel_row"],
            "Job ID": original.get("Job ID"),
            "Compound Name": original.get("Compound Name"),
            "SMILES": r["raw_smiles"],
            "canonical_smiles": r["canonical_smiles"],
            "APPf service model": original.get("Model"),
            "Status": original.get("Status"),
            "Submitted At": original.get("Submitted At"),
            "valid": r["valid"],
            "error": r["error"],
        }
        if r["valid"]:
            s = score_map[r["canonical_smiles"]]
            base.update({
                "priority_score": s["priority_score"],
                "raw_score": s["raw_score"],
                "bag_sd": s["bag_sd"],
                "above_threshold": bool(s["above_threshold"]),
                "in_domain": bool(s["in_domain"]),
                "max_train_tanimoto": s["max_train_tanimoto"],
                "rank_unique": s["rank_unique"],
                "rank_raw": s["rank_raw"],
                "rank_calibrated": s["rank_calibrated"],
                "in_merged_dataset": bool(s["in_merged_dataset"]),
                "merged_label_status": s["merged_label_status"],
                "merged_label": s["merged_label"],
                "mol_weight": s["mol_weight"],
                "heavy_atoms": s["heavy_atoms"],
                "fragment_like": bool(s["fragment_like"]),
                "modeling_split": s["modeling_split"],
                "in_training": bool(s["in_training"]),
            })
        else:
            base.update({
                "priority_score": np.nan, "raw_score": np.nan,
                "bag_sd": np.nan, "above_threshold": False,
                "in_domain": False, "max_train_tanimoto": np.nan,
                "rank_unique": np.nan, "rank_raw": np.nan,
                "rank_calibrated": np.nan, "in_merged_dataset": False,
                "merged_label_status": "", "merged_label": np.nan,
                "mol_weight": np.nan, "heavy_atoms": np.nan,
                "fragment_like": False, "modeling_split": "",
                "in_training": False,
            })
        out_rows.append(base)
    all_df = pd.DataFrame(out_rows)
    # Add counts back to the unique table.
    counts = all_df[all_df["valid"]].groupby("canonical_smiles").agg(
        n_rows=("excel_row", "size"), n_jobs=("Job ID", "nunique"),
    )
    unique = unique.set_index("canonical_smiles")
    unique["n_rows"] = counts["n_rows"].reindex(unique.index).fillna(0).astype(int)
    unique["n_jobs"] = counts["n_jobs"].reindex(unique.index).fillna(0).astype(int)
    unique = unique.reset_index()

    valid_unique = unique[unique["priority_score"].notna()].copy()
    top = valid_unique.sort_values("raw_score", ascending=False).head(50)
    top_recommended = valid_unique[
        valid_unique["above_threshold"]
        & valid_unique["in_domain"]
        & (~valid_unique["fragment_like"])
    ].sort_values("raw_score", ascending=False).head(50).copy()
    top_recommended.insert(
        0, "rank_recommended", np.arange(1, len(top_recommended) + 1)
    )
    invalid = all_df[~all_df["valid"]].copy()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PRED / f"excel_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_csv = out_dir / "all_rows.csv"
    unique_csv = out_dir / "unique_structures.csv"
    top_csv = out_dir / "top50_by_raw_score.csv"
    rec_csv = out_dir / "top50_recommended.csv"
    xlsx = out_dir / "predictions.xlsx"
    all_df.to_csv(all_csv, index=False, encoding="utf-8-sig")
    unique.to_csv(unique_csv, index=False, encoding="utf-8-sig")
    top.to_csv(top_csv, index=False, encoding="utf-8-sig")
    top_recommended.to_csv(rec_csv, index=False, encoding="utf-8-sig")

    summary = {
        "source_file": str(path),
        "rows_total": int(len(all_df)),
        "rows_valid": int(all_df["valid"].sum()),
        "rows_missing_smiles": int(all_df["SMILES"].isna().sum()),
        "rows_invalid": int((~all_df["valid"]).sum()),
        "unique_valid_structures": int(len(unique)),
        "unique_above_threshold": int(unique["above_threshold"].sum()),
        "unique_in_domain": int(unique["in_domain"].sum()),
        "unique_fragment_like": int(unique["fragment_like"].sum()),
        "unique_above_threshold_in_domain_not_fragment": int(
            (unique["above_threshold"] & unique["in_domain"]
             & (~unique["fragment_like"])).sum()
        ),
        "unique_in_merged_dataset": int(unique["in_merged_dataset"].sum()),
        "unique_in_training": int(unique["in_training"].sum()),
        "threshold": threshold,
        "ad_threshold": ad_threshold,
        "model": "EXT, 25 balanced CatBoost, split42",
        "runtime_utc": datetime.utcnow().isoformat(timespec="seconds"),
    }
    summary_df = pd.DataFrame(
        [{"metric": k, "value": v} for k, v in summary.items()]
    )
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        all_df.to_excel(writer, sheet_name="all_rows", index=False)
        unique.to_excel(writer, sheet_name="unique", index=False)
        top.to_excel(writer, sheet_name="top50", index=False)
        top_recommended.to_excel(
            writer, sheet_name="top50_recommended", index=False
        )
        invalid.to_excel(writer, sheet_name="invalid", index=False)
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        for ws in writer.book.worksheets:
            ws.freeze_panes = "A2"
            for col in ws.columns:
                letter = col[0].column_letter
                maxlen = max(
                    len(str(c.value)) if c.value is not None else 0
                    for c in col[: min(len(col), 500)]
                )
                ws.column_dimensions[letter].width = min(max(12, maxlen + 2), 70)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("wrote", xlsx)
    print("wrote", all_csv)
    print("wrote", unique_csv)
    print("wrote", top_csv)
    print("wrote", rec_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
