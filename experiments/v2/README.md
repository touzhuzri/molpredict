# V2 实验产物

本目录由 `scripts/run_v2_experiments.py` 生成，未纳入版本管理时可删除后重跑。

```
experiments/v2/
├── cache/                       # 可重建缓存
│   ├── smiles_states.pt         # ChemBERTa token 表示（fp16）
│   ├── smiles_tokens.pt         # ChemBERTa input_ids / mask（F-ft 用）
│   ├── confseq_states.pt        # 官方 ConfSeq token 表示
│   ├── confseq_tokens.pt
│   ├── padel24.csv              # APPf Table S2 的 24 个 PaDEL 描述符
│   ├── padel24_meta.json        # 缺失/跳过记录
│   ├── padel24_order.json       # 对齐校验用 InChIKey 列表
│   ├── rdkit217.npy             # RDKit 217 描述符
│   └── rdkit217_meta.json
├── splits/
│   ├── split_seed42.json        # train/val/test 索引
│   ├── split_seed43.json
│   └── split_seed44.json
├── runs/
│   ├── A_catboost_seed*/        # model.cbm, config.json, metrics.json, predictions_test.csv
│   ├── B_none_seed*/
│   ├── C_none_seed*/
│   ├── D_none_seed*/
│   ├── E_{attention,gated}_seed*/
│   ├── F_{attention,gated}_seed*/
│   ├── G_{attention,gated}_seed*/
│   └── F_attention_ft_seed*/    # 部分解冻；model.pt 含完整编码器（约 170MB/个）
├── summary.csv                  # 所有配置逐 seed 指标
├── stability.json               # 按 (experiment, fusion) 汇总 mean/std
├── summary_ag.csv / stability_ag.json  # 仅 A–G 的备份（不含 F-ft）
└── run_*.log
```

重建：

```bash
./.venv/Scripts/python.exe scripts/run_v2_experiments.py --prepare
./.venv/Scripts/python.exe scripts/run_v2_experiments.py \
  --experiments A,B,C,D,E,F,G --seeds 42,43,44 \
  --fusions attention,gated --epochs 60 --patience 12
```

`summary.csv` 在单独运行 `--finetune` 时会与已有行合并，不会丢掉 A–G。
