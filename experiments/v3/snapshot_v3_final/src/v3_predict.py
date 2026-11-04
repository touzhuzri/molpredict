"""V3 独立预测接口。

输入必须包含 SMILES、ConfSeq 字符串和 24 项 PaDEL 描述符（列名固定）。
输出是 PPDB 杀菌剂分类模型的分数，不是实验抗真菌活性概率。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence

from molpredict.padel import PADEL_NAMES
from molpredict.v3_data import NumericStandardizer
from molpredict.v3_encoders import (
    CONFSEQ_PAD_ID,
    encode_confseq_states,
    encode_smiles_states,
    load_confseq_parts,
    load_smiles_encoder,
    load_smiles_tokenizer,
    tokenize_confseq,
)
from molpredict.v3_model import V3Config
from molpredict.v3_train import load_model_for_inference


class V3Predictor:
    def __init__(self, model_dir: Path, device: str = "cpu"):
        self.model_dir = Path(model_dir)
        self.device = device
        self.feature_order = json.loads(
            (self.model_dir / "feature_order.json").read_text(encoding="utf-8")
        )["padel24"]
        if self.feature_order != PADEL_NAMES:
            raise ValueError("checkpoint 的 PaDEL 特征顺序与代码不一致")
        self.standardizer = NumericStandardizer.from_dict(
            json.loads((self.model_dir / "standardization.json").read_text(
                encoding="utf-8"))
        )
        self.model, self.config = load_model_for_inference(
            self.model_dir, device=device
        )
        self.threshold = float(
            json.loads((self.model_dir / "config.json").read_text(
                encoding="utf-8")).get("threshold", 0.5)
        )

        self.tokenizer = None
        self.smiles_encoder = None
        self.confseq_encoder = None
        self.confseq_vocab = None
        self.confseq_pad = CONFSEQ_PAD_ID
        self.confseq_unk = 3
        if "smiles" in self.config.modalities:
            self.tokenizer = load_smiles_tokenizer()
            if not self.config.smiles_live:
                self.smiles_encoder = load_smiles_encoder(0)
        if "confseq" in self.config.modalities:
            vocab_path = self.model_dir / "confseq_vocab.json"
            if vocab_path.exists():
                payload = json.loads(vocab_path.read_text(encoding="utf-8"))
                self.confseq_vocab = payload["vocab"]
                self.confseq_pad = int(payload["pad_id"])
                self.confseq_unk = int(payload["unk_id"])
            if not self.config.confseq_live:
                encoder, vocab, pad, unk = load_confseq_parts(0)
                self.confseq_encoder = encoder
                if self.confseq_vocab is None:
                    self.confseq_vocab = vocab
                self.confseq_pad = pad
                self.confseq_unk = unk

    def _encode_smiles(self, smiles, batch_size=32):
        if self.config.smiles_live:
            return None
        states, _, _, _ = encode_smiles_states(
            smiles, self.tokenizer, self.smiles_encoder,
            max_length=self.config.max_smiles_len, batch_size=batch_size,
        )
        return states

    def _encode_confseq(self, seqs, batch_size=32):
        if self.config.confseq_live:
            return None
        states, _, _ = encode_confseq_states(
            seqs, self.confseq_encoder, self.confseq_vocab,
            pad_id=self.confseq_pad, unk_id=self.confseq_unk,
            max_len=self.config.max_confseq_len,
            batch_size=batch_size,
        )
        return states

    def predict_frame(self, frame: pd.DataFrame,
                      threshold: float | None = None) -> pd.DataFrame:
        required = []
        if "smiles" in self.config.modalities:
            required.append("smiles")
        if "confseq" in self.config.modalities:
            required.append("confseq")
        if "descriptors" in self.config.modalities:
            required.extend(PADEL_NAMES)
        missing = [c for c in required if c not in frame.columns]
        if missing:
            raise ValueError(f"输入缺少列: {missing}")
        if len(frame) == 0:
            raise ValueError("输入为空")
        smiles = (frame["smiles"].astype(str).tolist()
                  if "smiles" in self.config.modalities else [])
        seqs = (frame["confseq"].astype(str).tolist()
                if "confseq" in self.config.modalities else [])
        if "descriptors" in self.config.modalities:
            descriptors = frame[PADEL_NAMES].to_numpy(dtype=np.float64)
            descriptors = self.standardizer.transform(descriptors)
        else:
            descriptors = np.zeros((len(frame), len(PADEL_NAMES)),
                                   dtype=np.float32)
        threshold = self.threshold if threshold is None else float(threshold)

        smiles_states = (self._encode_smiles(smiles)
                         if "smiles" in self.config.modalities else None)
        confseq_states = (self._encode_confseq(seqs)
                          if "confseq" in self.config.modalities else None)
        probs = []
        batch_size = 16
        for start in range(0, len(frame), batch_size):
            end = min(start + batch_size, len(frame))
            batch = {
                "descriptors": torch.tensor(
                    descriptors[start:end], dtype=torch.float32
                ),
            }
            if smiles_states is not None:
                items = smiles_states[start:end]
                mask = pad_sequence(
                    [torch.ones(x.size(0), dtype=torch.bool) for x in items],
                    batch_first=True, padding_value=False,
                )
                batch["smiles_states"] = pad_sequence(
                    items, batch_first=True, padding_value=0.0
                ).float()
                batch["smiles_mask"] = mask
            elif self.config.smiles_live:
                enc = self.tokenizer(
                    smiles[start:end], padding=True, truncation=True,
                    max_length=self.config.max_smiles_len,
                    return_tensors="pt",
                )
                batch["smiles_ids"] = (
                    enc["input_ids"], enc["attention_mask"].bool()
                )
            if confseq_states is not None:
                items = confseq_states[start:end]
                mask = pad_sequence(
                    [torch.ones(x.size(0), dtype=torch.bool) for x in items],
                    batch_first=True, padding_value=False,
                )
                batch["confseq_states"] = pad_sequence(
                    items, batch_first=True, padding_value=0.0
                ).float()
                batch["confseq_mask"] = mask
            elif self.config.confseq_live:
                ids = tokenize_confseq(
                    seqs[start:end], self.confseq_vocab,
                    max_len=self.config.max_confseq_len,
                    unk_id=self.confseq_unk,
                )
                padded = pad_sequence(
                    ids, batch_first=True, padding_value=self.confseq_pad
                )
                batch["confseq_ids"] = (
                    padded, (padded != self.confseq_pad)
                )
            with torch.no_grad():
                logits = self.model(batch)
                probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())

        out = frame.copy()
        out["probability"] = probs
        out["threshold"] = threshold
        out["predicted"] = (out["probability"] >= threshold).astype(int)
        return out


def predict_csv(model_dir: Path, input_csv: Path, out_csv: Path,
                threshold: float | None = None) -> pd.DataFrame:
    predictor = V3Predictor(model_dir)
    frame = pd.read_csv(input_csv)
    result = predictor.predict_frame(frame, threshold=threshold)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return result
