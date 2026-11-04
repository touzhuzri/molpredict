"""Compute/cache ChemBERTa token states and 24 PaDEL descriptors.

The encoder is the external pretrained ChemBERTa model used by the previous
project.  No old trained classifier weights are loaded.  PaDEL values are
computed with the official PaDEL-Descriptor jar through padelpy.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import DATA, FEATURES, dump_json, sha256  # noqa: E402
from molpredict.padel import PADEL_NAMES, compute_padel24  # noqa: E402
from molpredict.v3_encoders import (  # noqa: E402
    load_smiles_encoder, load_smiles_tokenizer,
)

MAX_SMILES_LEN = 512


def encode_states(smiles: list[str], tokenizer, encoder, device: str,
                  max_length: int = MAX_SMILES_LEN,
                  batch_size: int = 32):
    states, ids_list, masks, truncated = [], [], [], 0
    encoder.eval()
    use_amp = device == "cuda"
    with torch.no_grad():
        for start in range(0, len(smiles), batch_size):
            chunk = smiles[start:start + batch_size]
            enc = tokenizer(
                chunk, padding=True, truncation=True, max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    hidden = encoder(**enc).last_hidden_state
            else:
                hidden = encoder(**enc).last_hidden_state
            hidden = hidden.float()
            for j, text in enumerate(chunk):
                length = int(enc["attention_mask"][j].sum())
                n_raw = len(tokenizer(text, truncation=False)["input_ids"])
                if n_raw > max_length:
                    truncated += 1
                states.append(hidden[j, :length].to(torch.float16).cpu())
                ids_list.append(enc["input_ids"][j, :length].cpu())
                masks.append(enc["attention_mask"][j, :length].cpu())
    return states, ids_list, masks, truncated


def cache_smiles(struct: pd.DataFrame) -> dict:
    states_path = FEATURES / "smiles_states.pt"
    tokens_path = FEATURES / "smiles_tokens.pt"
    meta_path = FEATURES / "smiles_meta.json"
    ids = struct["structure_id"].astype(str).tolist()
    if states_path.exists() and tokens_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("structure_ids") == ids:
            payload = torch.load(states_path, weights_only=False)
            tokens = torch.load(tokens_path, weights_only=False)
            if tokens.get("structure_ids") == ids:
                return meta
    FEATURES.mkdir(parents=True, exist_ok=True)
    print(f"encoding {len(struct)} SMILES with ChemBERTa (max={MAX_SMILES_LEN})")
    tokenizer = load_smiles_tokenizer()
    encoder = load_smiles_encoder(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = encoder.to(device)
    started = time.time()
    states, token_ids, masks, truncated = encode_states(
        struct["canonical_smiles"].astype(str).tolist(), tokenizer, encoder,
        device=device, max_length=MAX_SMILES_LEN, batch_size=32,
    )
    torch.save({"states": states, "structure_ids": ids}, states_path)
    torch.save({"input_ids": token_ids, "attention_mask": masks,
                "structure_ids": ids}, tokens_path)
    meta = {
        "model": "seyonec/ChemBERTa-zinc-base-v1",
        "max_length": MAX_SMILES_LEN,
        "n": len(ids),
        "max_observed": int(max(len(x) for x in token_ids)),
        "truncated": int(truncated),
        "elapsed_sec": round(time.time() - started, 1),
        "device": device,
        "structure_ids": ids,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2)[:800])
    return meta


def cache_padel(struct: pd.DataFrame) -> dict:
    cache_dir = FEATURES / "padel"
    out = compute_padel24(
        struct["canonical_smiles"].astype(str).tolist(), cache_dir
    )
    df = pd.read_csv(out)
    if len(df) != len(struct):
        raise ValueError("PaDEL row count does not match structures")
    if df["smiles"].astype(str).tolist() != struct["canonical_smiles"].astype(str).tolist():
        raise ValueError("PaDEL SMILES order does not match structures")
    matrix = df[PADEL_NAMES].to_numpy(dtype=np.float32)
    np.save(FEATURES / "padel24.npy", matrix)
    meta_path = cache_dir / "padel24_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return {
        "csv": str(out),
        "csv_sha256": sha256(out),
        "nan_counts": {n: int(np.isnan(matrix[:, i]).sum())
                       for i, n in enumerate(PADEL_NAMES)},
        "rows_with_any_nan": int(np.isnan(matrix).any(axis=1).sum()),
        "skipped_molecules": len(meta.get("skipped_molecules", [])),
        "max_atoms": meta.get("max_atoms"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-padel", action="store_true")
    args = parser.parse_args()
    struct = pd.read_csv(DATA / "structures.csv", low_memory=False)
    struct = struct.sort_values("structure_index").reset_index(drop=True)
    if struct["structure_index"].tolist() != list(range(len(struct))):
        raise ValueError("structures.csv index is not contiguous")
    smiles_meta = cache_smiles(struct)
    payload = {
        "n_structures": int(len(struct)),
        "smiles": smiles_meta,
        "padel": None,
    }
    if not args.skip_padel:
        payload["padel"] = cache_padel(struct)
    dump_json(FEATURES / "feature_manifest.json", payload)
    print("feature manifest:", json.dumps(payload, ensure_ascii=False)[:1200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
