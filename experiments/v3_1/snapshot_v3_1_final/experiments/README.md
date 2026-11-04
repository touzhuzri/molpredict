# V3.1 实验产物

V3.1 保留 V3 的统一 token 序列（`[CLS] + SMILES + ConfSeq + 24 descriptors`），
在 GPU 上使用 BF16 混合精度、warmup+cosine、梯度累积和更长的早停耐心，并测试更深的
融合网络。所有结果写入本目录，不覆盖 `experiments/v3/`。

## 目录

```text
experiments/v3_1/
├── audit/                      # V3 学习行为审计（JSON + PNG）
├── group1/                     # 结构：base / L / M / D
├── group2/                     # 微调：ft_smiles / ft_both（最佳结构）
├── group3/                     # 消融：no_smiles / no_confseq / no_padel
├── smiles_check/               # 补充：ft_no_smiles（微调 ConfSeq、去掉 SMILES）
├── external/                   # 21 个外部已知分子的 ConfSeq/PaDEL/预测
├── gpu_check.json              # CUDA/AMP/模块设备实测
├── arch_params.json            # 各结构参数量
├── arch_params_live.json       # 含预训练编码器的参数量（如适用）
└── README.md
```

每个 run 目录包含：

```text
V3_1_<实验>_seed<N>/
├── model.pt
├── config.json                 # 结构、训练配方、设备、参数量、代码哈希
├── history.json                # 逐 epoch 损失/AUC/PR/lr/梯度/显存/耗时
├── metrics.json                # train/val/test 与注意力摘要
├── attention.json              # 按层、按头的跨模态注意力
├── predictions_test.csv
├── standardization.json
├── feature_order.json
├── split.json
├── pretrained.json
├── tokenizer/                  # 使用 SMILES 时
└── confseq_vocab.json          # 使用 ConfSeq 时
```

## 重建

```bash
# GPU 环境核验
./.venv/Scripts/python.exe scripts/check_v3_1_gpu.py

# 结构实验
./.venv/Scripts/python.exe scripts/run_v3_1_experiments.py \
  --experiments base,L,M,D --seeds 42,43,44 --epochs 100 --patience 18 \
  --batch-size 16 --grad-accum 2 --amp bf16 --out experiments/v3_1/group1

# 微调实验（结构由 --arch 指定；按验证集 one-SE + 最简规则选 base）
./.venv/Scripts/python.exe scripts/run_v3_1_experiments.py \
  --experiments ft_smiles,ft_both --arch base --seeds 42,43,44 \
  --epochs 100 --patience 18 --batch-size 16 --grad-accum 2 \
  --amp bf16 --out experiments/v3_1/group2

# 消融
./.venv/Scripts/python.exe scripts/run_v3_1_experiments.py \
  --experiments no_smiles,no_confseq,no_padel --arch base --seeds 42,43,44 \
  --epochs 100 --patience 18 --batch-size 16 --grad-accum 2 \
  --amp bf16 --out experiments/v3_1/group3
```

外部验证（21 个未参与训练的已知分子；输入需先用官方 ConfSeq/PaDEL 生成）：

```bash
./.venv/Scripts/python.exe scripts/external_validate_v3_1.py \
  --input experiments/v3_1/external/external_features.csv \
  --out experiments/v3_1/external --device cuda
```

结果见 `external/external_predictions.csv` 与 `external/external_summary.json`；
V3.1 外部 AUC 0.462–0.673，低于 CatBoost 0.750，不能用于实验筛选。
