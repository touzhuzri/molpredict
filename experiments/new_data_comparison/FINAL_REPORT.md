# FINAL_REPORT — 新数据辅助训练与五组模型对照

日期：2026-11-04
旧项目：`D:\works\yuce`
新数据（只读）：`D:\works\yuce\shujv\pesticide-data\data\processed\d2_1_training`

> 主任务标签是旧项目 PPDB 杀菌剂分类，**不是实验抗真菌活性**。
> 新数据用途标签是登记/产品用途证据，也不是实验活性。本实验没有把它们当作
> 旧任务标签，UNKNOWN 没有当作负样本，多成分产品证据没有分配给单个成分。

## 0. 执行摘要

- 新数据在**严格隔离**后只有 199 个可用于辅助学习的结构（MAIN 45），不是
  1,889 个：其中 1,130 行 / 900 个唯一 InChIKey 已在旧项目中，426 行是
  MULTI_COMPONENT，390 个 exact 新结构中 191 个又与旧 val/test 的 scaffold
  重叠。新数据没有负标签，因此没有做用途标签辅助分类。
- **新数据没有帮助旧主任务**：`D_OLD_ONLY` 0.7497 vs `D_NEW_AUX_MAIN`
  0.7175（−0.0322）、`D_NEW_AUX_FULL` 0.7332（−0.0165）。
- **24 项 PaDEL 没有提供可测量的有效帮助**：C（SMILES+PaDEL）0.7292 vs
  A（SMILES）0.7339，差 −0.0048；B（仅 PaDEL）只有 0.5723；C 的测试期
  PaDEL 置换几乎不改变 AUC（三种子平均下降 0.00015）。
- **辅助学习是唯一有正向迹象的改动**：`D_OLD_ONLY` 0.7497 vs C 0.7292，
  AUC +0.0206、PR-AUC +0.0749、MCC +0.0280；但 3 个 seed 中 2 个提高、
  1 个下降，标准差 0.028，不能称为稳健提升。
- **加深到 8 层没有帮助**：E 0.7200 vs C 0.7292（−0.0091）。
- 所有新配置都远低于历史 CatBoost A（0.8578）和 V3.1 `ft_both`（0.8342）；
  新实验的 A 组不是 V3.1，因为 A 不用 ConfSeq。
- 最值得继续的不是“更多新数据”，而是：保留简单 SMILES 基线，谨慎研究
  辅助学习，并把主要精力放在数据/标签质量与真正的外部验证上。

## 1. 旧项目审计

详见 `INITIAL_AUDIT.md`。核心事实：

- 旧数据 1,823 分子（431 阳性 / 1,392 阴性），PPDB 分类标签。
- 使用 V2 的 Murcko scaffold 划分 seed 42/43/44；训练/验证/测试分别为
  1332/168/323、1320/167/336、1295/208/320。
- V3.1 确实使用 SMILES token states、官方 ConfSeq token states 和 24 个
  PaDEL descriptor token；24 项特征进入共同注意力，不是只挂在旁边。
- 24 项特征名称、顺序、计算和标准化见 `INITIAL_AUDIT.md` 第 2.2 节；
  使用 padelpy 0.1.17 内置 PaDEL-Descriptor 的 2D 计算，未用 RDKit 替代。
- 历史参考（不是本次新结果）：CatBoost A 0.8578 ± 0.0292；V3.1 `base`
  0.7718 ± 0.0312；V3.1 `ft_both` 0.8342 ± 0.0307。

## 2. 新数据审计

详见 `INITIAL_AUDIT.md`、`data_audit/README.md`、`data_audit/new_aux_audit.json`。

| 项目 | 实际值 |
|---|---:|
| D2.1 全部候选 | 4,765 |
| 可解析结构 | 2,924 |
| VERIFIED / PROVISIONAL | 1 / 2,923 |
| AMBIGUOUS / NO_STRUCTURE | 185 / 1,656 |
| training_eligible | 1,889（MAIN 670 / EXTENDED 1,219） |
| 唯一 InChIKey | 1,529 |
| 已在旧项目中的唯一 InChIKey | 900 |
| MULTI_COMPONENT（eligible 内） | 426 |
| 排除旧分子、多组分、去重后的新结构 | 390（MAIN 79 / EXTENDED 311） |
| 再排除旧 val/test scaffold 重叠后的严格集 | **199（MAIN 45 / EXTENDED 154）** |

标签：10 个用途列只有 `POSITIVE` 或 `UNKNOWN`；明确负样本 0、冲突 0。
`label_masks=1` 只表示已知正标签；UNKNOWN 不是 0。

## 3. 实验设计

| 组 | 输入 | 数据条件 | 说明 |
| --- | --- | --- | --- |
| A | SMILES | OLD_ONLY | 结构基线 |
| B | 24 PaDEL | OLD_ONLY | 数值基线 |
| C | SMILES + 24 PaDEL | OLD_ONLY | 特征融合 |
| D | SMILES + 24 PaDEL + aux | OLD_ONLY / NEW_AUX_MAIN / NEW_AUX_FULL | 辅助学习 |
| E | SMILES + 24 PaDEL | OLD_ONLY | 8 层 vs C 的 4 层 |

- A/B/C/E 的 NEW_AUX 条件 N/A：没有辅助任务，把新数据加入主任务会改变主任务
  数据或标签语义。详见 `metrics/na_experiments.md`。
- D 的辅助任务：从 **只读取 SMILES 表示** 的池化向量预测 24 个标准化 PaDEL
  （masked MSE）；辅助头不看 PaDEL token，避免“抄输入”。aux_weight=0.2；
  主任务损失和辅助损失分别记录在 `history.json`。
- 主任务：旧 PPDB 标签、旧 scaffold 划分 seed 42/43/44、同一训练预算。
- 训练配方：AdamW，lr=3e-4，weight_decay=1e-4，batch 16×2 梯度累积，
  warmup 10% + cosine 到 0.1×，最多 100 epoch，验证 ROC-AUC 早停 patience=18，
  BF16，grad clip 5.0；阈值验证集最大 F1；测试集只评估一次。
- 标准化只在旧训练集拟合；新数据只贡献结构→PaDEL 辅助目标，不贡献旧任务标签。

### 3.1 隔离

- 新辅助集排除旧项目任意 InChIKey（全部 1,823 个，包括 train/val/test）。
- 严格主分析再排除旧 val/test 的 Murcko scaffold 重叠。
- 旧项目 test/val 不进入辅助训练；旧项目 test 不参与任何模型选择。
- 新数据 390 个 exact 结构中 37 个至少一个 PaDEL 缺失；辅助 loss 用 mask 排除，
  不用 0 代替。

## 4. 结果

所有指标为旧项目三套 scaffold 划分的测试集，n=3，均值 ± 样本标准差。
图：`figures/auc_summary.png`。

| 实验 | 输入/条件 | 测试 AUC | 测试 PR-AUC | 测试 F1 | 测试 MCC | 验证 AUC | 训练 AUC | 训练-测试 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `A_OLD_ONLY` | SMILES | 0.7339 ± 0.0241 | 0.4377 ± 0.0375 | 0.4609 ± 0.0945 | 0.2752 ± 0.0869 | 0.8169 ± 0.0577 | 0.9939 ± 0.0074 | 0.2600 ± 0.0225 |
| `B_OLD_ONLY` | 24 PaDEL | 0.5723 ± 0.0307 | 0.3188 ± 0.0312 | 0.3197 ± 0.0672 | 0.0585 ± 0.0250 | 0.6740 ± 0.0513 | 0.5551 ± 0.0125 | −0.0172 ± 0.0312 |
| `C_OLD_ONLY` | SMILES + PaDEL | 0.7292 ± 0.0206 | 0.4643 ± 0.0261 | 0.5075 ± 0.0165 | 0.3176 ± 0.0328 | 0.8024 ± 0.0747 | 0.9962 ± 0.0042 | 0.2670 ± 0.0213 |
| `D_OLD_ONLY` | C + aux（旧训练集） | **0.7497 ± 0.0374** | **0.5392 ± 0.0667** | **0.5282 ± 0.0440** | **0.3456 ± 0.0552** | 0.8506 ± 0.0563 | 0.9972 ± 0.0021 | 0.2475 ± 0.0390 |
| `D_NEW_AUX_MAIN` | C + aux（+45 新结构） | 0.7175 ± 0.0331 | 0.4265 ± 0.0684 | 0.4997 ± 0.0130 | 0.2966 ± 0.0162 | 0.8051 ± 0.0539 | 0.9933 ± 0.0074 | 0.2757 ± 0.0375 |
| `D_NEW_AUX_FULL` | C + aux（+199 新结构） | 0.7332 ± 0.0391 | 0.4693 ± 0.0401 | 0.5073 ± 0.0309 | 0.3168 ± 0.0523 | 0.8032 ± 0.0665 | 0.9858 ± 0.0171 | 0.2526 ± 0.0289 |
| `E_OLD_ONLY` | C 的输入，8 层 | 0.7200 ± 0.0154 | 0.4671 ± 0.0443 | 0.4976 ± 0.0528 | 0.3277 ± 0.0470 | 0.7959 ± 0.0940 | 0.9976 ± 0.0017 | 0.2776 ± 0.0162 |

训练成本（同一台 RTX 4060 Laptop，batch 16×2，BF16）：

| 实验 | 参数量 | 每 epoch 秒 | 峰值分配显存 | 单 run 平均耗时 |
| --- | ---: | ---: | ---: | ---: |
| A | 678,913 | 1.39 | 149 MB | 70 s |
| B | 550,785 | 1.28 | 33 MB | 46 s |
| C | 690,433 | 1.43 | 156 MB | 80 s |
| D（三种条件） | 700,505 | 1.3–1.4 | 166–171 MB | 62–80 s |
| E（8 层） | 1,220,353 | 1.72 | 254 MB | 98 s |

逐 seed 测试 AUC：

| 实验 | seed42 | seed43 | seed44 | 参数量 | 最佳 epoch |
| --- | ---: | ---: | ---: | ---: | --- |
| `A_OLD_ONLY` | 0.7148 | 0.7610 | 0.7259 | 678,913 | 31/39/21 |
| `B_OLD_ONLY` | 0.5392 | 0.6000 | 0.5776 | 550,785 | 8/5/40 |
| `C_OLD_ONLY` | 0.7058 | 0.7369 | 0.7447 | 690,433 | 33/21/53 |
| `D_OLD_ONLY` | 0.7277 | 0.7285 | 0.7929 | 700,505 | 51/32/26 |
| `D_NEW_AUX_MAIN` | 0.7402 | 0.6796 | 0.7329 | 700,505 | 19/33/38 |
| `D_NEW_AUX_FULL` | 0.7327 | 0.6944 | 0.7725 | 700,505 | 42/13/25 |
| `E_OLD_ONLY` | 0.7352 | 0.7044 | 0.7205 | 1,220,353 | 32/33/48 |

关键配对差（同 seed，均值 ± 标准差）：

| 对比 | ΔAUC | ΔPR-AUC | ΔF1 | ΔMCC |
| --- | --- | --- | --- | --- |
| C − A（加 PaDEL） | −0.0048 ± 0.0217 | +0.0266 ± 0.0252 | +0.0466 ± 0.0788 | +0.0424 ± 0.0717 |
| C − B（SMILES vs 仅 PaDEL） | +0.1569 ± 0.0173 | +0.1455 ± 0.0060 | +0.1878 ± 0.0575 | +0.2591 ± 0.0165 |
| D_OLD − C（辅助学习） | +0.0206 ± 0.0283 | +0.0749 ± 0.0644 | +0.0207 ± 0.0409 | +0.0280 ± 0.0702 |
| D_NEW_MAIN − D_OLD（新 MAIN） | −0.0322 ± 0.0391 | −0.1127 ± 0.0271 | −0.0285 ± 0.0342 | −0.0489 ± 0.0492 |
| D_NEW_FULL − D_OLD（新全量） | −0.0165 ± 0.0198 | −0.0699 ± 0.0303 | −0.0209 ± 0.0192 | −0.0288 ± 0.0277 |
| E − C（加深） | −0.0091 ± 0.0336 | +0.0028 ± 0.0220 | −0.0099 ± 0.0683 | +0.0102 ± 0.0541 |

观察：

- A/C/D/E 的训练 AUC 都到 0.98–0.999，训练-测试差 0.25–0.28；这是明显的
  scaffold 过拟合。B 训练 AUC 只有 0.555，模型几乎没学会，属于欠拟合。
- D_OLD 的 AUC/PR/F1/MCC 都是新实验中最好的；但只有 seed44 明显提高，
  seed42 小幅提高、seed43 小幅下降。不能把 +0.0206 当作稳健增益。
- 新数据加入辅助集后，MAIN 和 FULL 都低于 D_OLD；没有证据支持新数据有用。

### 4.1 与历史基线比较（同一旧 scaffold 划分，仅作参考）

| 模型 | 测试 AUC | 说明 |
| --- | ---: | --- |
| CatBoost A | 0.8578 ± 0.0292 | 旧最佳基线 |
| V3.1 `ft_both` | 0.8342 ± 0.0307 | 历史 V3.1，使用 ConfSeq |
| V3.1 `base` | 0.7718 ± 0.0312 | 历史 V3.1，使用 ConfSeq |
| 本次 `D_OLD_ONLY` | 0.7497 ± 0.0374 | 本次新实验中最好 |
| 本次 `A_OLD_ONLY` | 0.7339 ± 0.0241 | 本次最简单基线 |
| 本次 `E_OLD_ONLY` | 0.7200 ± 0.0154 | 8 层 |
| 本次 `B_OLD_ONLY` | 0.5723 ± 0.0307 | 仅 24 PaDEL |

历史结果来自 `experiments/v3_1/`（ConfSeq + SMILES + PaDEL），A–E 不含 ConfSeq，
因此不是严格单变量比较；这张表只说明“这批新配置没有接近旧最佳”。

## 5. 24 项 PaDEL 是否真正被使用

### 5.1 三组对照

- B（仅 PaDEL）0.5723，远低于 A（仅 SMILES）0.7339；B 的训练 AUC 0.555，
  说明这个 token 架构无法从 24 个标量学到有效分类边界，不是“PaDEL 很强”。
- C（SMILES+PaDEL）0.7292，低于 A 0.7339（−0.0048）；PR-AUC/F1/MCC 略高
  （+0.0266/+0.0466/+0.0424），但 AUC 没有提高。PaDEL 没有带来稳定的主任务收益。

### 5.2 测试期置换检验

图：`figures/padel_permutation.png`。

对 C 的三个 seed 模型，在测试集上只置换 PaDEL（SMILES、模型、划分不变）：

| seed | 基线 AUC | 行置换平均 AUC | 下降 | 列置换下降 |
| ---: | ---: | ---: | ---: | ---: |
| 42 | 0.7059 | 0.7051 | −0.0007（几乎为 0） | −0.0007 |
| 43 | 0.7369 | 0.7371 | +0.0002（无下降） | +0.0004 |
| 44 | 0.7448 | 0.7448 | +0.0001（无下降） | +0.0001 |

三种子平均：行置换 AUC 下降 **0.00015**，列置换下降 **0.00006**。
**C 模型基本没有使用 PaDEL 特征**；不能因为输入分支存在或梯度非零就宣称有效。

### 5.3 补充：简单 MLP

24 项 PaDEL 的简单 MLP（24→64→32→1，独立于 A–E 主矩阵）：

| seed | 测试 AUC | PR-AUC |
| ---: | ---: | ---: |
| 42 | 0.6107 | 0.3254 |
| 43 | 0.6285 | 0.3725 |
| 44 | 0.6245 | 0.3615 |
| 均值 | 0.6212 | 0.3531 |

即使换成更适合表格特征的 MLP，24 项 PaDEL 单独也只有 ~0.62；与历史 V2
PaDEL-only 0.634 一致。**24 项特征单独预测能力有限，融合模型也没有真正利用它们。**

## 6. 对五个问题的回答

**Q1：新数据是否真的帮助了旧模型？**
没有。`D_OLD_ONLY` 0.7497；加入严格隔离的新 MAIN（45 结构）后 0.7175
（−0.0322）；加入 MAIN+EXTENDED（199 结构）后 0.7332（−0.0165）。PR-AUC、
F1、MCC 也都下降。新数据在这个辅助任务上没有带来可测收益，反而略差。
可能原因：严格隔离后新结构太少；新结构多为 PROVISIONAL；PaDEL 辅助目标本身
与主任务弱相关；没有负标签，无法做用途标签辅助分类。不能据此断言“任何新数据
都无用”，但本次这批数据没有被证明有用。

**Q2：24 项 PaDEL 是否有效？**
没有证据支持有效。C vs A 的 AUC −0.0048；B 仅 0.5723；置换测试几乎不改变 AUC
（平均 −0.00015）；简单 MLP 也只有 0.6212。24 项描述符在这套架构和这套数据上
没有提供可靠的主任务增益。注意：这不等于“PaDEL 特征本身没有化学信息”，而是
“在当前模型/数据/融合方式下没有被利用”。

**Q3：辅助学习是否有效？**
有方向性但不够稳健的正向迹象。`D_OLD_ONLY` 比 C 高 AUC +0.0206、PR-AUC
+0.0749、MCC +0.0280；但 3 个 seed 中 2 个提高、1 个下降，标准差 0.028。
辅助任务本身确实在学习（aux loss 从 ~1.0 降到 ~0.7），但主任务收益不稳定。
可以认为“值得继续研究”，不能认为“已经证明有效”。

**Q4：更深的网络是否有效？**
没有。E（8 层，1.22 M 参数）0.7200，低于 C（4 层，0.69 M）0.7292，差
−0.0091；PR-AUC/F1/MCC 也没有一致提高。加深到 8 层没有带来收益，反而更慢。

**Q5：最终哪种方案最值得继续？**
按本次三 seed 的 AUC/PR/MCC，`D_OLD_ONLY`（0.7497）最好，但优势只有
+0.0206 AUC，且不稳定；A（0.7339）最简单、无辅助任务、无新数据依赖，是更
稳妥的基线。**没有一种新方案接近历史 CatBoost A（0.8578）或 V3.1
`ft_both`（0.8342）。** 如果继续投入，优先方向应是：
1. 提高数据/标签质量与 scaffold 覆盖，而不是继续加新数据或加层；
2. 若要研究辅助学习，把辅助目标设计得更贴近主任务，并用更多 seed/外部
   scaffold 验证 D_OLD 的正向迹象；
3. 若必须部署，仍使用 CatBoost A 或 V3.1 `ft_both`，而不是本次 A–E。

## 7. 公平性、泄漏与限制

- 主任务数据、划分、seed、优化器、训练预算、早停、阈值和评估对所有组一致。
- 特征标准化只在旧训练集拟合；新数据没有参与旧任务标准化。
- 旧项目任意 InChIKey 不进入新辅助集；严格分析还排除了旧 val/test scaffold。
- 旧 test 从不参与早停、阈值或模型选择；每个 run 的测试集只评估一次。
- 三套 scaffold 测试集彼此有重叠，mean±std 不是三次独立复现。
- 新辅助集严格隔离后只有 45/199 个结构，统计功效低；新数据只有正标签，
  无法做用途标签辅助分类；多成分证据没有分配。
- 新数据结构多数是 PROVISIONAL，不是人工 VERIFIED；PaDEL 有 NaN，用 mask 排除。
- 旧任务标签和新数据用途标签都不是实验活性；不能把结果解释成抗真菌活性。
- B 组是 token 架构下的数值基线；它欠拟合并不等于 PaDEL 特征无用，补充 MLP
  也只到 0.62，说明这批 24 项特征单独预测能力有限。

## 7.1 对旧代码的改动

`src/molpredict/v3_1_train.py` 增加了可选参数 `model`、`aux_loader`、`aux_weight`、
`aux_source`，并新增辅助 batch 的搬运/masked MSE；默认值保持原有 V3.1 行为不变。
`src/molpredict/new_data_model.py` 是新增文件。旧 V3.1 代码/结果快照仍在
`experiments/v3_1/snapshot_v3_1_final/`，旧 run 的权重和指标未被覆盖。
`src/molpredict/v3_predict.py` 的 GPU 设备搬运修复也保留。

## 8. 复现

```bash
cd D:/works/yuce

# 1) 构建严格隔离的新辅助集（只读新项目）
./.venv/Scripts/python.exe experiments/new_data_comparison/scripts/build_new_aux_data.py

# 2) 五组实验（A/B/C/E OLD_ONLY + D 三种辅助条件）
./.venv/Scripts/python.exe experiments/new_data_comparison/scripts/run_new_data_experiments.py \
  --experiments A_OLD_ONLY,B_OLD_ONLY,C_OLD_ONLY,D_OLD_ONLY,D_NEW_AUX_MAIN,D_NEW_AUX_FULL,E_OLD_ONLY \
  --seeds 42,43,44 --epochs 100 --patience 18 --batch-size 16 --grad-accum 2 \
  --aux-weight 0.2 --amp bf16 --out experiments/new_data_comparison

# 3) PaDEL 置换检验（C 的三个 seed）
for s in 42 43 44; do
  ./.venv/Scripts/python.exe experiments/new_data_comparison/scripts/padel_permutation.py \
    --model experiments/new_data_comparison/checkpoints/C_OLD_ONLY_seed$s \
    --out experiments/new_data_comparison/ablations/padel_permutation_seed$s.json \
    --repeats 32
done

# 4) 补充 MLP
./.venv/Scripts/python.exe experiments/new_data_comparison/scripts/padel_mlp_baseline.py

# 5) 汇总表
./.venv/Scripts/python.exe experiments/new_data_comparison/scripts/summarize_results.py
```

产物：`configs/*.yaml`、`checkpoints/*`、`metrics/summary.csv` 与 `stability.json`、
`ablations/`、`figures/`、`data_audit/`、`logs/`，以及本报告和 `INITIAL_AUDIT.md`。
