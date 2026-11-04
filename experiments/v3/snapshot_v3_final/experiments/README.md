# V3 实验产物

本目录由 `scripts/run_v3_experiments.py` 生成。V3 复用 V2 的 ConfSeq/PaDEL 缓存，
只新增 320-token 的 SMILES 缓存和 V3 运行产物。

当前正式汇总包含 13 个 run：`full` / `no_smiles` / `no_confseq` / `no_padel` 各
seed42/43/44，加上 `full_ft` 的 seed42。指标见 `summary.csv`，三划分稳定性见
`stability.json`（`full_ft` 的 n=1，std 为 0）。

```text
experiments/v3/
├── cache/
│   ├── smiles_states.pt       # ChemBERTa token 表示，max_length=320
│   ├── smiles_tokens.pt
│   ├── smiles_meta.json
│   └── manifest.json          # 训练 CSV/缓存哈希、PaDEL 顺序与 NaN 计数
├── splits/                    # 从 V2 复制的 scaffold 划分索引
├── runs/
│   └── V3_<实验>_seed<N>/
│       ├── model.pt           # 融合网络；full_ft 还含解冻的 ChemBERTa
│       ├── config.json        # 结构、训练参数、阈值、数据/缓存/预训练来源
│       ├── tokenizer/         # ChemBERTa tokenizer
│       ├── confseq_vocab.json # ConfSeq 词表、pad/unk、官方 checkpoint 哈希
│       ├── standardization.json
│       ├── feature_order.json
│       ├── split.json
│       ├── pretrained.json
│       ├── metrics.json
│       └── predictions_test.csv
├── leakage_audit.json         # 数据内泄漏审计；预训练重叠标为 unknown
├── summary.csv                # 逐 seed 指标
└── stability.json             # 按实验汇总 mean/std
```

`full_ft` 的 `model.pt` 含完整 ChemBERTa，约 170 MB/个；其余 cached run 的
`model.pt` 只有融合网络（约 3 MB）。

重建：

```bash
./.venv/Scripts/python.exe scripts/run_v3_experiments.py --prepare
./.venv/Scripts/python.exe scripts/run_v3_experiments.py \
  --experiments full,no_smiles,no_confseq,no_padel \
  --seeds 42,43,44 --epochs 40 --patience 8
```

如果并行启动不同配置（各自写独立 `--out`），用
`scripts/aggregate_v3_runs.py --roots ... --out experiments/v3` 合并。

预测（输入 CSV 必须含 `smiles`、`confseq` 和 24 个 PaDEL 列）：

```bash
molpredict v3-predict --model experiments/v3/runs/V3_full_seed42 \
  --input new_molecules.csv --out predictions.csv
```
