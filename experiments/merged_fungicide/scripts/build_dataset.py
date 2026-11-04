"""Build the merged fungicide record table, label evidence and audit.

Rules implemented:
- old PPDB labels keep their original meaning: 1 = fungicide category only,
  0 = herbicide/insecticide category only;
- a new FUNGICIDE=POSITIVE label is a use-evidence label, not experimental
  activity;
- new UNKNOWN is never converted to 0;
- records are kept, not deduplicated; structures are grouped only for training;
- a structure with both positive and negative evidence is CONFLICT and is not
  used in ordinary supervised training.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    DATA, NEW_ELIGIBLE, NEW_EVIDENCE, NEW_LABELS, NEW_MOLECULES,
    OLD_LABELED, OUT, TARGET_USES, canonicalize, dump_json, scaffold_key,
    sha256,
)


def _read_old() -> pd.DataFrame:
    old = pd.read_csv(OLD_LABELED)
    expected = {"name", "smiles", "label", "category", "source", "ppdb_id",
                "inchikey"}
    missing = expected - set(old.columns)
    if missing:
        raise ValueError(f"old PPDB file missing columns: {missing}")
    if not set(old["label"].unique()).issubset({0, 1}):
        raise ValueError("old PPDB label contains values other than 0/1")
    old = old.copy()
    old["record_id"] = "OLD_PPDB_" + old["ppdb_id"].astype(str)
    old["source_dataset"] = "OLD_PPDB"
    old["source_name"] = "PPDB"
    old["source_record_id"] = old["ppdb_id"].astype(str)
    old["original_name"] = old["name"]
    old["raw_smiles"] = old["smiles"]
    old["raw_label"] = old["label"].astype(int)
    old["raw_label_status"] = np.where(old["label"] == 1, "POSITIVE",
                                       "NEGATIVE")
    old["raw_category"] = old["category"]
    old["structure_form"] = "SINGLE_COMPONENT"
    old["structure_status"] = "OLD_PPDB"
    old["identity_status"] = "OLD_PPDB"
    old["eligibility_tier"] = "OLD"
    old["license_status"] = "OLD_PPDB_UNSPECIFIED"
    old["source_list"] = "PPDB"
    old["positive_use_categories"] = np.where(
        old["label"] == 1, "FUNGICIDE", ""
    )
    old["n_evidence_sources"] = 1
    return old


def _read_new() -> pd.DataFrame:
    new = pd.read_csv(NEW_MOLECULES, low_memory=False)
    labels = pd.read_csv(NEW_LABELS)
    eligible = pd.read_csv(NEW_ELIGIBLE)
    evidence = pd.read_csv(NEW_EVIDENCE, low_memory=False)

    if new["substance_id"].duplicated().any():
        raise ValueError("new molecules.csv has duplicate substance_id")
    new = new.merge(
        labels[["substance_id"] + TARGET_USES +
               [f"{u}_status" for u in TARGET_USES]],
        on="substance_id", how="left", validate="one_to_one",
    )
    new = new.merge(
        eligible[["substance_id", "eligibility_tier"]],
        on="substance_id", how="left",
    )
    new["eligibility_tier"] = new["eligibility_tier"].fillna("NOT_ELIGIBLE")

    use_cols = [u for u in TARGET_USES if u in new.columns]
    new["positive_use_categories"] = new[use_cols].apply(
        lambda row: ";".join(u for u in use_cols if row.get(u) == 1), axis=1
    )
    source_lists = evidence.groupby("substance_id")["evidence_source"].apply(
        lambda s: ";".join(sorted(set(s.astype(str))))
    )
    new["source_list"] = new["substance_id"].map(source_lists).fillna("")
    new["n_evidence_sources"] = new["source_list"].map(
        lambda s: 0 if not s else len(s.split(";"))
    )
    new["record_id"] = "NEW_" + new["substance_id"].astype(str)
    new["source_dataset"] = "NEW_D2_1"
    new["source_name"] = "pesticide-data D2.1 candidate"
    new["source_record_id"] = new["substance_id"].astype(str)
    new["original_name"] = new["canonical_name"]
    new["raw_smiles"] = new["smiles"]
    new["raw_label"] = new["FUNGICIDE_status"].map(
        {"POSITIVE": 1, "UNKNOWN": pd.NA}
    ).astype("Int64")
    # UNKNOWN is deliberately blank, not 0.  New data has no negative label.
    new["raw_label_status"] = new["FUNGICIDE_status"]
    new["raw_category"] = new["positive_use_categories"]
    new.loc[new["structure_form"].isna(), "structure_form"] = "NO_STRUCTURE"
    new["structure_status"] = new["structure_status"].fillna("NO_STRUCTURE")
    new["identity_status"] = new["identity_status"].fillna("UNRESOLVED")
    new["license_status"] = new["license_status"].fillna("UNKNOWN")
    return new


def _combine_records(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "record_id", "source_dataset", "source_name", "source_record_id",
        "original_name", "raw_smiles", "inchikey", "structure_form",
        "structure_status", "identity_status", "eligibility_tier",
        "license_status", "raw_label", "raw_label_status", "raw_category",
        "positive_use_categories", "source_list", "n_evidence_sources",
    ]
    old = old.copy()
    new = new.copy()
    for frame in (old, new):
        for c in cols:
            if c not in frame:
                frame[c] = ""
    out = pd.concat([old[cols], new[cols]], ignore_index=True)
    out["canonical_smiles"] = out["raw_smiles"].map(canonicalize)
    out["has_structure"] = out["canonical_smiles"].notna().astype(int)
    out["structure_id"] = out["canonical_smiles"]
    out.loc[out["has_structure"] == 0, "structure_id"] = None
    out["structure_form"] = np.where(
        out["has_structure"] == 1, out["structure_form"], "NO_STRUCTURE"
    )
    return out


def _structure_table(records: pd.DataFrame) -> pd.DataFrame:
    structured = records[records["has_structure"] == 1].copy()
    grouped = structured.groupby("structure_id", sort=True)
    rows = []
    for sid, g in grouped:
        old_g = g[g["source_dataset"] == "OLD_PPDB"]
        new_g = g[g["source_dataset"] == "NEW_D2_1"]
        old_pos = bool((old_g["raw_label_status"] == "POSITIVE").any())
        old_neg = bool((old_g["raw_label_status"] == "NEGATIVE").any())
        new_pos = bool((new_g["raw_label_status"] == "POSITIVE").any())
        new_unk = bool((new_g["raw_label_status"] == "UNKNOWN").any())

        if old_pos and old_neg:
            status = "CONFLICT"
        elif old_neg and new_pos:
            status = "CONFLICT"
        elif old_pos:
            status = "POSITIVE"
        elif old_neg:
            status = "NEGATIVE"
        elif new_pos:
            status = "POSITIVE"
        elif new_unk:
            status = "UNKNOWN"
        else:
            status = "UNKNOWN"

        forms = set(g["structure_form"].dropna().astype(str))
        form = "MULTI_COMPONENT" if "MULTI_COMPONENT" in forms else (
            "SINGLE_COMPONENT" if "SINGLE_COMPONENT" in forms else "UNKNOWN"
        )
        rows.append({
            "structure_id": sid,
            "canonical_smiles": sid,
            "scaffold": scaffold_key(sid),
            "structure_form": form,
            "is_multi_component": int(form == "MULTI_COMPONENT"),
            "n_records": len(g),
            "n_old_records": len(old_g),
            "n_new_records": len(new_g),
            "old_positive": int(old_pos),
            "old_negative": int(old_neg),
            "new_positive": int(new_pos),
            "new_unknown": int(new_unk),
            "n_old_positive_records": int(
                (old_g["raw_label_status"] == "POSITIVE").sum()
            ),
            "n_old_negative_records": int(
                (old_g["raw_label_status"] == "NEGATIVE").sum()
            ),
            "n_new_positive_records": int(
                (new_g["raw_label_status"] == "POSITIVE").sum()
            ),
            "label_status": status,
            "label": (1 if status == "POSITIVE" else
                      0 if status == "NEGATIVE" else np.nan),
            "model_eligible": int(status in {"POSITIVE", "NEGATIVE"}),
            "representative_record_id": g["record_id"].iloc[0],
            "old_categories": ";".join(sorted(set(
                old_g["raw_category"].dropna().astype(str).tolist()
            ))),
            "new_tiers": ";".join(sorted(set(
                new_g["eligibility_tier"].dropna().astype(str).tolist()
            ))),
            "new_sources": ";".join(sorted({
                s for v in new_g["source_list"].fillna("").astype(str)
                for s in v.split(";") if s
            })),
        })
    struct = pd.DataFrame(rows).sort_values("structure_id").reset_index(drop=True)
    struct["structure_index"] = np.arange(len(struct), dtype=int)
    return struct


def _apply_structure_labels(records: pd.DataFrame,
                            struct: pd.DataFrame) -> pd.DataFrame:
    info = struct.set_index("structure_id")[
        ["label_status", "label", "model_eligible", "scaffold",
         "structure_form", "structure_index"]
    ]
    out = records.copy()
    for col in info.columns:
        out[col] = out["structure_id"].map(info[col])
    out["label_status"] = out["label_status"].fillna("NO_STRUCTURE")
    out["structure_form"] = out["structure_form"].fillna("NO_STRUCTURE")
    out["model_exclusion_reason"] = ""
    out.loc[out["has_structure"] == 0, "model_exclusion_reason"] = (
        "NO_PARSEABLE_STRUCTURE"
    )
    out.loc[out["label_status"] == "UNKNOWN", "model_exclusion_reason"] = (
        "UNKNOWN_LABEL"
    )
    out.loc[out["label_status"] == "CONFLICT", "model_exclusion_reason"] = (
        "LABEL_CONFLICT"
    )
    out["model_eligible"] = out["model_eligible"].fillna(0).astype(int)
    return out


def _label_evidence(records: pd.DataFrame) -> pd.DataFrame:
    old = records[records["source_dataset"] == "OLD_PPDB"].copy()
    old_rows = pd.DataFrame({
        "evidence_id": "OLD_EVID_" + old["record_id"],
        "record_id": old["record_id"],
        "structure_id": old["structure_id"],
        "source_dataset": old["source_dataset"],
        "source": "PPDB",
        "source_record_id": old["source_record_id"],
        "use_category": np.where(old["raw_label"] == 1, "FUNGICIDE",
                                 old["raw_category"]),
        "evidence_status": np.where(
            old["raw_label"] == 1, "POSITIVE",
            "NEGATIVE_IN_PPDB_CATEGORY"
        ),
        "evidence_type": "PPDB_CATEGORY_MEMBERSHIP",
        "attribution_level": "CATEGORY_LEVEL",
        "original_use_text": old["raw_category"].astype(str),
        "product_id": "",
        "contributes_to_label": 1,
        "label_value": old["raw_label"].astype(int),
        "license_status": old["license_status"],
        "note": ("PPDB list membership; 0 means present only in the "
                 "herbicide/insecticide lists, not an experimental "
                 "non-fungicide assay."),
    })

    ev = pd.read_csv(NEW_EVIDENCE, low_memory=False)
    ev = ev[ev["use_category"] == "FUNGICIDE"].copy()
    ev = ev.merge(
        records.loc[records["source_dataset"] == "NEW_D2_1",
                    ["source_record_id", "record_id", "structure_id"]],
        left_on="substance_id", right_on="source_record_id", how="left",
        validate="many_to_one",
    )
    new_rows = pd.DataFrame({
        "evidence_id": "NEW_EVID_" + ev.index.astype(str),
        "record_id": ev["record_id"],
        "structure_id": ev["structure_id"],
        "source_dataset": "NEW_D2_1",
        "source": ev["evidence_source"],
        "source_record_id": ev["substance_id"],
        "use_category": ev["use_category"],
        "evidence_status": ev["evidence_status"],
        "evidence_type": ev["evidence_type"],
        "attribution_level": ev["attribution_level"],
        "original_use_text": ev["original_use_text"],
        "product_id": ev["product_id"],
        "contributes_to_label": ev["contributes_to_label"],
        "label_value": 1,
        "license_status": ev["license_status"],
        "note": ("Use/registration evidence; not an experimental activity "
                 "measurement. Product-level rows do not contribute to the "
                 "candidate label."),
    })
    return pd.concat([old_rows, new_rows], ignore_index=True)


def _write_outputs(records: pd.DataFrame, struct: pd.DataFrame,
                   evidence: pd.DataFrame) -> dict:
    DATA.mkdir(parents=True, exist_ok=True)
    record_cols = [
        "record_id", "source_dataset", "source_name", "source_record_id",
        "original_name", "raw_smiles", "canonical_smiles", "inchikey",
        "has_structure", "structure_form", "structure_status",
        "identity_status", "eligibility_tier", "license_status",
        "raw_label", "raw_label_status", "raw_category",
        "positive_use_categories", "source_list", "n_evidence_sources",
        "structure_id", "structure_index", "scaffold", "label_status",
        "label", "model_eligible", "model_exclusion_reason",
    ]
    records[record_cols].to_csv(
        DATA / "merged_fungicide_records.csv", index=False, encoding="utf-8-sig"
    )
    struct.to_csv(DATA / "structures.csv", index=False, encoding="utf-8-sig")
    evidence.to_csv(
        DATA / "label_evidence.csv", index=False, encoding="utf-8-sig"
    )

    unknown = records[
        (records["label_status"] == "UNKNOWN")
        | ((records["has_structure"] == 0)
           & (records["raw_label_status"] == "UNKNOWN"))
    ].copy()
    unknown.to_csv(
        DATA / "unknown_records.csv", index=False, encoding="utf-8-sig"
    )
    conflicts = records[records["label_status"] == "CONFLICT"].copy()
    conflicts.to_csv(
        DATA / "conflicts.csv", index=False, encoding="utf-8-sig"
    )

    # Extra diagnostic: positives whose structure could not be parsed.
    unusable_pos = records[
        (records["raw_label_status"] == "POSITIVE")
        & (records["has_structure"] == 0)
    ].copy()
    unusable_pos.to_csv(
        DATA / "positive_without_structure.csv", index=False,
        encoding="utf-8-sig",
    )

    # Positive/negative/unknown summary at record and structure level.
    stats = {
        "old_source_file": str(OLD_LABELED),
        "old_source_sha256": sha256(OLD_LABELED),
        "new_source_files": {
            "molecules": str(NEW_MOLECULES),
            "molecules_sha256": sha256(NEW_MOLECULES),
            "labels": str(NEW_LABELS),
            "labels_sha256": sha256(NEW_LABELS),
            "evidence": str(NEW_EVIDENCE),
            "evidence_sha256": sha256(NEW_EVIDENCE),
        },
        "old_records": int((records["source_dataset"] == "OLD_PPDB").sum()),
        "new_records": int((records["source_dataset"] == "NEW_D2_1").sum()),
        "total_records": int(len(records)),
        "old_parseable_records": int((
            (records["source_dataset"] == "OLD_PPDB")
            & (records["has_structure"] == 1)
        ).sum()),
        "new_parseable_records": int((
            (records["source_dataset"] == "NEW_D2_1")
            & (records["has_structure"] == 1)
        ).sum()),
        "parseable_records": int((records["has_structure"] == 1).sum()),
        "unique_structures": int(len(struct)),
        "old_only_structures": int((
            (struct["n_old_records"] > 0) & (struct["n_new_records"] == 0)
        ).sum()),
        "new_only_structures": int((
            (struct["n_old_records"] == 0) & (struct["n_new_records"] > 0)
        ).sum()),
        "both_source_structures": int((
            (struct["n_old_records"] > 0) & (struct["n_new_records"] > 0)
        ).sum()),
        "old_positive_structures": int((struct["old_positive"] == 1).sum()),
        "new_positive_structures": int((struct["new_positive"] == 1).sum()),
        "duplicate_parseable_records": int(
            (records["has_structure"] == 1).sum() - len(struct)
        ),
        "structure_duplicate_groups": int(
            (struct["n_records"] > 1).sum()
        ),
        "old_label_counts": {
            str(k): int(v) for k, v in
            records.loc[records["source_dataset"] == "OLD_PPDB",
                        "raw_label_status"].value_counts().items()
        },
        "new_label_counts": {
            str(k): int(v) for k, v in
            records.loc[records["source_dataset"] == "NEW_D2_1",
                        "raw_label_status"].value_counts().items()
        },
        "structure_label_counts": {
            str(k): int(v) for k, v in
            struct["label_status"].value_counts().items()
        },
        "trainable_structures": int(struct["model_eligible"].sum()),
        "trainable_positive": int((struct["label"] == 1).sum()),
        "trainable_negative": int((struct["label"] == 0).sum()),
        "conflict_structures": int((struct["label_status"] == "CONFLICT").sum()),
        "conflict_records": int(len(conflicts)),
        "unknown_records": int(len(unknown)),
        "unknown_structures": int((struct["label_status"] == "UNKNOWN").sum()),
        "raw_unknown_records": int(
            (records["raw_label_status"] == "UNKNOWN").sum()
        ),
        "unknown_no_structure_records": int((
            (records["has_structure"] == 0)
            & (records["raw_label_status"] == "UNKNOWN")
        ).sum()),
        "unknown_on_known_negative_structure_records": int((
            (records["raw_label_status"] == "UNKNOWN")
            & (records["label_status"] == "NEGATIVE")
        ).sum()),
        "positive_records_without_structure": int(len(unusable_pos)),
        "multi_component_records": int((
            records["structure_form"] == "MULTI_COMPONENT"
        ).sum()),
        "no_structure_records": int((
            records["structure_form"] == "NO_STRUCTURE"
        ).sum()),
        "single_component_records": int((
            records["structure_form"] == "SINGLE_COMPONENT"
        ).sum()),
        "trainable_multi_component": int((
            (struct["model_eligible"] == 1)
            & (struct["is_multi_component"] == 1)
        ).sum()),
        "trainable_positive_multi_component": int((
            (struct["model_eligible"] == 1) & (struct["label"] == 1)
            & (struct["is_multi_component"] == 1)
        ).sum()),
        "old_only_positive_structures": int((
            (struct["label"] == 1) & (struct["old_positive"] == 1)
            & (struct["new_positive"] == 0)
        ).sum()),
        "both_positive_structures": int((
            (struct["label"] == 1) & (struct["old_positive"] == 1)
            & (struct["new_positive"] == 1)
        ).sum()),
        "new_only_positive_structures": int((
            (struct["label"] == 1) & (struct["old_positive"] == 0)
            & (struct["new_positive"] == 1)
        ).sum()),
        "negative_structures_with_new_records": int((
            (struct["label"] == 0) & (struct["n_new_records"] > 0)
        ).sum()),
        "new_tier_record_counts": {
            str(k): int(v) for k, v in records.loc[
                records["source_dataset"] == "NEW_D2_1",
                "eligibility_tier"
            ].value_counts().items()
        },
        "new_tier_positive_structure_counts": {
            "MAIN": int(((struct["label"] == 1)
                         & struct["new_tiers"].str.contains("MAIN")).sum()),
            "EXTENDED_ONLY": int(((struct["label"] == 1)
                                  & struct["new_tiers"].str.contains("EXTENDED")
                                  & ~struct["new_tiers"].str.contains("MAIN")).sum()),
        },
        "label_evidence_rows": int(len(evidence)),
        "label_evidence_contributing_rows": int(
            evidence["contributes_to_label"].sum()
        ),
        "outputs": {
            "merged_records": "data/merged_fungicide_records.csv",
            "structures": "data/structures.csv",
            "label_evidence": "data/label_evidence.csv",
            "unknown_records": "data/unknown_records.csv",
            "conflicts": "data/conflicts.csv",
            "positive_without_structure":
                "data/positive_without_structure.csv",
        },
    }
    dump_json(DATA / "dataset_statistics.json", stats)
    return stats


def main() -> int:
    old = _read_old()
    new = _read_new()
    records = _combine_records(old, new)
    struct = _structure_table(records)
    records = _apply_structure_labels(records, struct)
    evidence = _label_evidence(records)
    stats = _write_outputs(records, struct, evidence)
    print(f"records={len(records)} unique_structures={len(struct)}")
    print(f"trainable structures={struct['model_eligible'].sum()} "
          f"pos={stats['trainable_positive']} "
          f"neg={stats['trainable_negative']}")
    print(f"conflict structures={stats['conflict_structures']} "
          f"conflict records={stats['conflict_records']} "
          f"unknown records={stats['unknown_records']}")
    print(f"unparsed positives {stats['positive_records_without_structure']}")
    print(f"outputs in {OUT / 'data'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
