"""Local Streamlit demo for the APPf-like fungicide-likeness model.

Run from the repository root:
    ./.venv/Scripts/python.exe -m streamlit run \
        experiments/merged_fungicide/scripts/demo_app.py

The app is a research prototype.  It ranks structures by documented
fungicide-use evidence; it does not measure antifungal activity.
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from catboost import CatBoostClassifier, Pool
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from appf_features import (  # noqa: E402
    MORGAN_N_BITS, feature_names, maccs_fingerprint, morgan_fingerprint,
    rdkit_descriptors,
)
from common import canonicalize  # noqa: E402
from molpredict.padel import PADEL_NAMES, compute_padel24  # noqa: E402

RDLogger.DisableLog("rdApp.*")

APPF = ROOT / "experiments" / "merged_fungicide" / "appf_like"
MODEL_DIR = APPF / "models" / "EXT" / "split42"
PRED = APPF / "predictions"
FEAT = APPF / "features"
FIGS = APPF / "figures"
SUMMARY = APPF / "metrics" / "summary.json"
FEATURE_NAMES = feature_names("EXT")
THRESHOLD = 0.4615
AD_FLAG = 0.30
CACHE = APPF / "demo_cache"


def load_models() -> list[CatBoostClassifier]:
    paths = sorted(MODEL_DIR.glob("bag*.cbm"))
    if not paths:
        raise FileNotFoundError(f"no models in {MODEL_DIR}")
    models = []
    for p in paths:
        m = CatBoostClassifier()
        m.load_model(str(p))
        models.append(m)
    return models


@st.cache_resource
def cached_models() -> list[CatBoostClassifier]:
    return load_models()


@st.cache_resource
def cached_train_fp() -> np.ndarray:
    return np.load(FEAT / "train_morgan_split42.npy")


@st.cache_resource
def cached_calibrator() -> IsotonicRegression:
    val = pd.read_csv(PRED / "EXT_split42_val.csv")
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(val["prob_raw"].to_numpy(float), val["label"].to_numpy(int))
    return iso


@st.cache_data
def cached_summary() -> dict:
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


@st.cache_data
def cached_test_examples() -> pd.DataFrame:
    return pd.read_csv(PRED / "EXT_split42_test_with_ad.csv")


def parse_text(text: str) -> list[dict]:
    rows = []
    for i, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name = f"Molecule {i}"
        smiles = line
        if "\t" in line:
            name, smiles = line.split("\t", 1)
        elif "," in line:
            left, right = line.split(",", 1)
            if left.strip() and right.strip():
                name, smiles = left.strip(), right.strip()
        rows.append({"name": name.strip(), "smiles": smiles.strip()})
    return rows


def parse_upload(upload) -> list[dict]:
    name = upload.name.lower()
    data = upload.getvalue()
    rows = []
    if name.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(data))
        cols = {c.lower(): c for c in df.columns}
        if "smiles" not in cols:
            raise ValueError("CSV must contain a 'smiles' column")
        smi_col = cols["smiles"]
        name_col = cols.get("name") or cols.get("id")
        for i, row in df.iterrows():
            rows.append({
                "name": str(row[name_col]) if name_col else f"Row {i + 1}",
                "smiles": str(row[smi_col]),
            })
    elif name.endswith(".sdf"):
        with tempfile.NamedTemporaryFile(suffix=".sdf", delete=False) as f:
            f.write(data)
            path = f.name
        try:
            supplier = Chem.SDMolSupplier(path)
            for i, mol in enumerate(supplier):
                if mol is None:
                    continue
                nm = mol.GetProp("_Name") if mol.HasProp("_Name") else f"Mol {i + 1}"
                rows.append({"name": nm, "smiles": Chem.MolToSmiles(mol)})
        finally:
            Path(path).unlink(missing_ok=True)
    else:
        raise ValueError("Upload a .csv or .sdf file")
    return rows


def padel_for(records: list[dict]) -> np.ndarray:
    valid = [r for r in records if r["mol"] is not None]
    n = len(records)
    out = np.full((n, len(PADEL_NAMES)), np.nan, dtype=np.float32)
    if not valid:
        return out
    CACHE.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="padel_", dir=CACHE) as tmp:
        try:
            csv_path = compute_padel24(
                [r["canonical"] for r in valid], Path(tmp)
            )
            df = pd.read_csv(csv_path).sort_values("order")
            vals = df[PADEL_NAMES].apply(
                pd.to_numeric, errors="coerce"
            ).to_numpy(float)
            # Map valid rows back to the original positions.
            valid_pos = [i for i, r in enumerate(records) if r["mol"] is not None]
            out[valid_pos] = vals
        except Exception as exc:  # PaDEL can fail for unusual structures
            st.warning(f"PaDEL descriptors unavailable: {exc}")
    return out


def featurize(records: list[dict]) -> np.ndarray:
    mols = [r["mol"] for r in records]
    rd = np.vstack([
        rdkit_descriptors(m) if m is not None else np.full(12, np.nan)
        for m in mols
    ]).astype(np.float32)
    morgan = np.vstack([
        morgan_fingerprint(m) if m is not None else np.zeros(MORGAN_N_BITS,
                                                             dtype=np.uint8)
        for m in mols
    ]).astype(np.uint8)
    maccs = np.vstack([
        maccs_fingerprint(m) if m is not None else np.zeros(167, dtype=np.uint8)
        for m in mols
    ]).astype(np.uint8)
    padel = padel_for(records)
    X = np.concatenate([padel, rd, morgan, maccs], axis=1).astype(np.float32)
    X[~np.isfinite(X)] = np.nan
    return X


def max_tanimoto(morgan_rows: np.ndarray,
                 train: np.ndarray) -> np.ndarray:
    def to_fp(row):
        fp = DataStructs.ExplicitBitVect(MORGAN_N_BITS)
        fp.SetBitsFromList(np.flatnonzero(row).tolist())
        return fp

    train_fps = [to_fp(r) for r in train]
    return np.asarray([
        max(DataStructs.BulkTanimotoSimilarity(to_fp(q), train_fps))
        for q in morgan_rows
    ])


def predict(records: list[dict]) -> tuple[pd.DataFrame, np.ndarray]:
    models = cached_models()
    iso = cached_calibrator()
    X = featurize(records)
    raw = np.mean([
        m.predict_proba(X)[:, 1] for m in models
    ], axis=0)
    cal = iso.predict(raw)
    morgan = np.vstack([
        morgan_fingerprint(r["mol"]) if r["mol"] is not None
        else np.zeros(MORGAN_N_BITS, dtype=np.uint8)
        for r in records
    ])
    maxsim = max_tanimoto(morgan, cached_train_fp())
    out = pd.DataFrame({
        "name": [r["name"] for r in records],
        "smiles": [r["canonical"] if r["canonical"] else r["smiles"]
                   for r in records],
        "priority_score": cal,
        "raw_score": raw,
        "in_domain": maxsim >= AD_FLAG,
        "max_train_tanimoto": maxsim,
        "above_threshold": cal >= THRESHOLD,
        "valid": [r["mol"] is not None for r in records],
    }).sort_values("priority_score", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out, X


def shap_table(row: np.ndarray, model: CatBoostClassifier,
               top_n: int = 10) -> pd.DataFrame:
    pool = Pool(row.reshape(1, -1), feature_names=FEATURE_NAMES)
    sv = model.get_feature_importance(data=pool, type="ShapValues")[0, :-1]
    idx = np.argsort(-np.abs(sv))[:top_n]
    return pd.DataFrame({
        "feature": [FEATURE_NAMES[i] for i in idx],
        "contribution": sv[idx],
        "value": row[idx],
    })


st.set_page_config(
    page_title="Fungicide-likeness demo",
    page_icon="🌿",
    layout="wide",
)
st.title("Fungicide-likeness prioritisation demo")
st.caption(
    "APPf-style CatBoost prototype · 24 PaDEL + RDKit + Morgan/MACCS · "
    "25 balanced models · scaffold split 42"
)
st.warning(
    "Research prototype. The score is a **priority / fungicide-likeness** "
    "score based on documented use labels, not a measured antifungal activity "
    "and not a validated probability. Do not use it as the sole basis for "
    "synthesis or biological claims."
)

with st.sidebar:
    st.header("Model card")
    try:
        summ = cached_summary()
        sel = summ["split42_selected"]["metrics"]
        st.metric("Test ROC-AUC (split 42)", f"{sel['roc_auc']:.3f}")
        st.metric("Test PR-AUC", f"{sel['pr_auc']:.3f}")
        st.metric("Test F1 at validation threshold", f"{sel['f1']:.3f}")
        st.metric("Top-10% precision", f"{sel['topk']['top_10pct']['precision']:.3f}")
        ext_mean = summ["catboost"]["EXT"]
        st.caption(
            f"n = {sel['n']} structures ({sel['positives']} positive). "
            f"Threshold = {THRESHOLD:.3f}. "
            f"3-split mean AUC = {ext_mean['test_auc_mean']:.3f} ± "
            f"{ext_mean['test_auc_std']:.3f}."
        )
    except Exception as exc:
        st.error(f"Could not load summary: {exc}")
    st.divider()
    st.markdown(
        "**Flags**\n\n"
        f"- Score ≥ {THRESHOLD:.3f}: above the validation-selected threshold\n"
        f"- Tanimoto to training < {AD_FLAG:.2f}: outside the applicability "
        "domain (lower confidence)\n"
        "- `bag_sd`: disagreement across the 25 balanced models\n"
    )
    st.divider()
    st.caption(
        "Data: old PPDB categories + new D2.1 registration/use evidence. "
        "UNKNOWN and CONFLICT labels are not treated as negatives."
    )

tab_predict, tab_card, tab_help = st.tabs(
    ["Score molecules", "Model performance", "How to read the output"]
)

with tab_predict:
    st.subheader("Input")
    mode = st.radio(
        "Input mode", ["Paste SMILES", "Upload CSV", "Upload SDF"],
        horizontal=True,
    )
    records_in = []
    if mode == "Paste SMILES":
        examples = pd.DataFrame([
            ["Tebuconazole", "CC(C)(C)C(O)(CCc1ccc(Cl)cc1)Cn1cncn1"],
            ["Carbendazim", "COC(=O)Nc1nc2ccccc2[nH]1"],
            ["Azoxystrobin",
             "CO/C=C(/C(=O)OC)c1ccccc1Oc1cc(Oc2ccccc2C#N)ncn1"],
            ["Chlorothalonil", "N#Cc1c(Cl)c(Cl)c(Cl)c(C#N)c1Cl"],
            ["Atrazine", "CCNc1nc(Cl)nc(NC(C)C)n1"],
        ], columns=["name", "smiles"])
        with st.expander("Example molecules (illustrative; some may be in training)"):
            st.dataframe(examples, hide_index=True, width="stretch")
            if st.button("Load these examples"):
                st.session_state["smiles_text"] = "\n".join(
                    f"{r['name']}\t{r['smiles']}" for _, r in examples.iterrows()
                )
            st.caption(
                "For an unbiased quick check, load a random sample of held-out "
                "test structures instead. Names show the true label only for "
                "the demo; the model does not see labels."
            )
            if st.button("Load 12 random held-out test molecules"):
                tdf = cached_test_examples()
                sample = tdf.sample(n=min(12, len(tdf)), random_state=None)
                lines = []
                for i, (_, row) in enumerate(sample.iterrows(), start=1):
                    tag = "positive" if int(row["label"]) == 1 else "negative"
                    lines.append(
                        f"held-out {tag} #{i}\t{row['canonical_smiles']}"
                    )
                st.session_state["smiles_text"] = "\n".join(lines)
        text = st.text_area(
            "One molecule per line: `SMILES` or `name<TAB>SMILES`",
            key="smiles_text",
            height=180,
            placeholder="Tebuconazole\tCC(C)(C)C(O)(Cc1ccc(Cl)cc1)n1cncn1",
        )
        if text.strip():
            records_in = parse_text(text)
    elif mode == "Upload CSV":
        up = st.file_uploader("CSV with a `smiles` column", type=["csv"])
        if up is not None:
            try:
                records_in = parse_upload(up)
            except Exception as exc:
                st.error(str(exc))
    else:
        up = st.file_uploader("SDF file", type=["sdf"])
        if up is not None:
            try:
                records_in = parse_upload(up)
            except Exception as exc:
                st.error(str(exc))

    if records_in:
        # Canonicalize and flag invalid entries.
        for r in records_in:
            r["canonical"] = canonicalize(r["smiles"])
            r["mol"] = Chem.MolFromSmiles(r["canonical"]) if r["canonical"] else None
        invalid = [r for r in records_in if r["mol"] is None]
        if invalid:
            st.warning(
                f"{len(invalid)} invalid SMILES were skipped: "
                + "; ".join(r["smiles"] for r in invalid[:5])
            )
        records_in = [r for r in records_in if r["mol"] is not None]
        if records_in:
            st.caption(
                f"{len(records_in)} valid molecules. PaDEL calculation may "
                "take a few seconds per batch."
            )
            if st.button("Score molecules", type="primary"):
                with st.spinner("Calculating descriptors and scoring..."):
                    try:
                        res, X = predict(records_in)
                        st.session_state["results"] = res
                        st.session_state["records"] = records_in
                        st.session_state["X"] = X
                    except Exception as exc:
                        st.exception(exc)

    res = st.session_state.get("results")
    if res is not None:
        st.subheader("Ranked results")
        show = res[[
            "rank", "name", "smiles", "priority_score", "raw_score",
            "above_threshold", "in_domain", "max_train_tanimoto",
        ]].copy()
        show["priority_score"] = show["priority_score"].round(3)
        show["raw_score"] = show["raw_score"].round(3)
        show["max_train_tanimoto"] = show["max_train_tanimoto"].round(3)
        st.dataframe(
            show,
            hide_index=True,
            width="stretch",
            column_config={
                "priority_score": st.column_config.ProgressColumn(
                    "priority score", min_value=0.0, max_value=1.0,
                    format="%.3f",
                ),
                "raw_score": st.column_config.NumberColumn(
                    "raw score", format="%.3f"
                ),
                "above_threshold": st.column_config.CheckboxColumn(
                    "above threshold"
                ),
                "in_domain": st.column_config.CheckboxColumn(
                    "in domain"
                ),
                "max_train_tanimoto": st.column_config.NumberColumn(
                    "max Tanimoto", format="%.3f"
                ),
            },
        )
        st.download_button(
            "Download results CSV",
            res.to_csv(index=False).encode("utf-8"),
            "fungicide_likeness_scores.csv",
            "text/csv",
        )
        st.info(
            "Scores are for prioritisation only. The model was trained on "
            "documented fungicide-use labels, not on a uniform antifungal "
            "assay. Out-of-domain and high-`bag_sd` molecules should be "
            "treated as lower-confidence hypotheses."
        )

        # Per-molecule explanation.
        st.subheader("Why did a molecule score this way?")
        sel_name = st.selectbox(
            "Molecule", res["name"].tolist(),
            index=0, key="explain_molecule",
        )
        row = res.index[res["name"] == sel_name][0]
        si = int(row)
        try:
            model = cached_models()[0]
            X = st.session_state["X"]
            shap = shap_table(X[si], model, top_n=10)
            st.caption(
                "SHAP contributions (log-odds) from one representative "
                "balanced model. Positive values push toward fungicide-like."
            )
            chart = shap.set_index("feature")["contribution"].sort_values()
            st.bar_chart(chart, horizontal=True, height=340)
            st.dataframe(shap, hide_index=True, width="stretch")
        except Exception as exc:
            st.warning(f"Could not compute SHAP explanation: {exc}")

with tab_card:
    st.subheader("What was measured")
    try:
        summ = cached_summary()
        rows = []
        for name, m in summ["catboost"].items():
            rows.append({
                "model": name,
                "test AUC (mean ± SD)": f"{m['test_auc_mean']:.3f} ± "
                                       f"{m['test_auc_std']:.3f}",
                "test PR-AUC": f"{m['test_pr_auc_mean']:.3f}",
                "test F1": f"{m['test_f1_mean']:.3f}",
                "test MCC": f"{m['test_mcc_mean']:.3f}",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True,
                     width="stretch")
        st.markdown(
            "- **P24** is the faithful APPf-style replication: 24 PaDEL "
            "descriptors + CatBoost + 25 balanced subsets.\n"
            "- **EXT** adds 12 RDKit descriptors, Morgan(2048) and MACCS(167); "
            "it is the demo model.\n"
            "- **EXT_OLD_ONLY** removes all new-only-positive training "
            "structures, used as a source-shift check."
        )
        st.image(str(FIGS / "model_comparison.png"),
                 caption="CatBoost vs the existing deep models")
        st.image(str(FIGS / "roc_pr_ext.png"),
                 caption="Selected EXT model, scaffold test split 42")
        c1, c2 = st.columns(2)
        c1.image(str(FIGS / "calibration_ext.png"),
                 caption="Validation-fitted isotonic calibration")
        c2.image(str(FIGS / "score_distribution.png"),
                 caption="Test score distribution")
        st.image(str(FIGS / "applicability_domain.png"),
                 caption="Applicability domain; flag at Tanimoto 0.30")
        st.image(str(FIGS / "source_shift.png"),
                 caption="Old-only training -> new-only positives")
        st.image(str(FIGS / "permutation_null.png"),
                 caption="Label-permutation null test")
        st.image(str(FIGS / "chemical_space.png"),
                 caption="Morgan chemical space by scaffold split")
    except Exception as exc:
        st.error(f"Could not load performance figures: {exc}")

with tab_help:
    st.subheader("What the score means")
    st.markdown(
        "The model is trained on **documented fungicide use / registration "
        "evidence** from old PPDB categories and new D2.1 labels. UNKNOWN and "
        "CONFLICT structures are not counted as negatives. Therefore the "
        "output is best read as:\n\n"
        "> a ranked hypothesis about whether a structure belongs to the "
        "chemical space of documented fungicides.\n\n"
        "It is **not**:\n\n"
        "- a measured antifungal activity;\n"
        "- a pathogen-specific prediction;\n"
        "- a calibrated clinical/field probability;\n"
        "- a substitute for a dose-response assay.\n\n"
        "### Recommended use in a lab\n\n"
        "1. Score a compound library.\n"
        "2. Prioritise the top 10–20% and check the applicability-domain "
        "flag.\n"
        "3. Test top-ranked compounds plus a random control set in the same "
        "pathogen panel.\n"
        "4. Feed the assay results back as labels; retrain and evaluate "
        "prospectively.\n\n"
        "### What would make this a real activity model\n\n"
        "- 300–500 compounds tested under one protocol;\n"
        "- 3–5 pathogens, with explicit inactive compounds;\n"
        "- an independent blinded external set;\n"
        "- expert adjudication of ambiguous use labels."
    )