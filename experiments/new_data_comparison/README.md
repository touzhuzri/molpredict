# new_data_comparison

用新项目 `shujv/pesticide-data/data/processed/d2_1_training` 做一轮严格对照，
回答：新数据、24 项 PaDEL、辅助学习、网络深度分别有没有价值。

新项目**只读**；所有输出在本目录，旧项目 V2/V3/V3.1 的产物未覆盖。

## 目录

```text
experiments/new_data_comparison/
├── INITIAL_AUDIT.md       # 旧项目 + 新数据审计（实验前）
├── FINAL_REPORT.md        # 最终报告与五个问题的回答
├── configs/               # A–E 每个实验的 YAML
├── data_audit/            # 新数据过滤链、候选表、数据哈希
├── data_aux/              # 新辅助集 new_aux.pt（SMILES states + PaDEL）
├── splits/                # 旧 scaffold 划分的只读副本
├── scripts/               # 构建辅助集、训练、置换、MLP、汇总
├── checkpoints/           # 21 个 run（7 配置 × 3 seed）
├── logs/                  # 训练与置换日志
├── metrics/               # summary.csv、stability.json、tables.md
├── ablations/             # PaDEL 置换、补充 MLP
└── figures/               # AUC 汇总图、置换图
```

## 七个实际配置

- `A_OLD_ONLY`：SMILES only
- `B_OLD_ONLY`：24 PaDEL only
- `C_OLD_ONLY`：SMILES + 24 PaDEL
- `D_OLD_ONLY`：C + structure→PaDEL 辅助任务（辅助集只用旧训练集）
- `D_NEW_AUX_MAIN`：C + aux（+严格隔离的新 MAIN，45 结构）
- `D_NEW_AUX_FULL`：C + aux（+严格隔离的 MAIN+EXTENDED，199 结构）
- `E_OLD_ONLY`：C 的输入，8 层 Transformer

A/B/C/E 的 NEW_AUX 条件为 N/A（没有辅助任务，无法公平使用新数据）；见
`metrics/na_experiments.md`。

## 关键结果（测试 AUC，三 seed 均值 ± 标准差）

| 配置 | AUC |
| --- | ---: |
| A_OLD_ONLY | 0.7339 ± 0.0241 |
| B_OLD_ONLY | 0.5723 ± 0.0307 |
| C_OLD_ONLY | 0.7292 ± 0.0206 |
| D_OLD_ONLY | 0.7497 ± 0.0374 |
| D_NEW_AUX_MAIN | 0.7175 ± 0.0331 |
| D_NEW_AUX_FULL | 0.7332 ± 0.0391 |
| E_OLD_ONLY | 0.7200 ± 0.0154 |

结论：新数据没有帮助；24 项 PaDEL 没有被 C 模型实际使用（置换下降 ~0.00015）；
辅助学习有弱的正向迹象但不稳健；8 层没有收益。历史 CatBoost A（0.8578）
和 V3.1 ft_both（0.8342）仍远高于这些新配置。

## 复现

```bash
cd D:/works/yuce
./.venv/Scripts/python.exe experiments/new_data_comparison/scripts/build_new_aux_data.py
./.venv/Scripts/python.exe experiments/new_data_comparison/scripts/run_new_data_experiments.py \
  --experiments A_OLD_ONLY,B_OLD_ONLY,C_OLD_ONLY,D_OLD_ONLY,D_NEW_AUX_MAIN,D_NEW_AUX_FULL,E_OLD_ONLY \
  --seeds 42,43,44 --epochs 100 --patience 18 --batch-size 16 --grad-accum 2 \
  --aux-weight 0.2 --amp bf16
./.venv/Scripts/python.exe experiments/new_data_comparison/scripts/padel_permutation.py \
  --model experiments/new_data_comparison/checkpoints/C_OLD_ONLY_seed42 \
  --out experiments/new_data_comparison/ablations/padel_permutation_seed42.json
./.venv/Scripts/python.exe experiments/new_data_comparison/scripts/padel_mlp_baseline.py
./.venv/Scripts/python.exe experiments/new_data_comparison/scripts/summarize_results.py
```

## 解释边界

- 主任务标签是 PPDB 分类，不是实验活性；新数据用途标签也不是实验活性。
- 三套 scaffold 测试集有重叠，mean±std 不是三次独立复现。
- 严格隔离后的新辅助结构只有 45/199，统计功效低；新数据无负标签，未做用途
  标签辅助分类。
- 实验过程中未修改新项目任何文件；数据哈希见 `data_audit/data_manifest.json`。
