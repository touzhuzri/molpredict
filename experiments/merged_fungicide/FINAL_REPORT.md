# 新旧数据合并的独立杀菌剂用途预测模型：最终报告

生成目录：`D:\works\yuce\experiments\merged_fungicide`
日期：2026-11-04

## 摘要

本报告把旧项目 PPDB 数据与新项目 `pesticide-data` D2.1 候选数据合并，建立
一个只预测“是否有杀菌剂用途证据”的独立模型。合并后保留 6,592 条记录、
3,100 个唯一可解析结构，其中 2,078 个结构有可监督标签（726 正 / 1,352 负）。
42 个结构因正负证据冲突被排除；UNKNOWN 不进入二分类损失，也没有转成 0。

训练了三个新模型：A（SMILES）、B（SMILES + 24 PaDEL）、C（B + 结构→PaDEL
辅助学习），并做了 B 的“只用旧标签训练”来源消融。所有模型随机初始化，不加载
旧项目分类器权重；测试集只在早停和阈值确定后评估一次。

主要结果：C 的平均测试 AUC 最高（0.7886 ± 0.0161），但相对 B 仅
+0.0068；A 为 0.7593；B_OLD_ONLY 为 0.7623。测试期 PaDEL 置换只下降
<0.002，说明 24 个描述符基本没有被主分类器使用。新数据对“仅新数据正”
子集有方向性帮助，但总体增益不稳健。详见第 6–7 节。

## 1. 实际路径与只读边界

- 旧项目实际根目录：`D:\works\yuce`。
- 任务描述中的 `D:\works\yuce\molpredict` 不存在；旧项目的 `data/`、`src/`、
  `experiments/` 位于 `D:\works\yuce` 根目录。没有寻找其他项目替代。
- 新数据：`D:\works\yuce\shujv\pesticide-data\data\processed\d2_1_training`。
- 新项目全程只读；没有修改、删除或覆盖任何新项目文件。
- 没有操作 CREA、其他化学项目或其他 Claude Code 窗口。
- 所有新代码、数据、模型、日志和报告只写在 `experiments/merged_fungicide/`。
- 旧模型、旧实验结果、旧原始数据和旧划分均未覆盖。

## 2. 数据流

```text
旧 PPDB (1,827 条)
  + 新 D2.1 (4,765 个候选)
        ↓ 规范 SMILES / 保留原始记录
6,592 条合并记录
        ↓ 结构分组，不去重记录
3,100 个唯一可解析结构
        ↓ 结构级标签规则
726 正 / 1,352 负 / 980 UNKNOWN / 42 CONFLICT
        ↓ 排除 UNKNOWN 与 CONFLICT
2,078 个可监督结构
        ↓ scaffold 分组
1,454 / 312 / 312（train/val/test）
```

逐项审计见 `reports/data_audit.md`。核心数字：

| 指标 | 数量 |
| --- | ---: |
| 旧记录 | 1,827 |
| 新数据上游来源记录（项目文档） | 约 133,404 |
| 新候选分子记录（实际使用） | 4,765 |
| 合并记录 | 6,592 |
| 可解析记录 | 4,751 |
| 唯一可解析结构 | 3,100 |
| 可监督结构 | 2,078 |
| 正 / 负 | 726 / 1,352 |
| 结构级 UNKNOWN | 980 |
| 结构级 CONFLICT | 42（110 条记录） |
| 无结构正记录 | 375 |
| 多成分可监督结构 | 260（正 168） |

## 3. 标签规则

### 3.1 旧数据

- `label=1`：只出现在 PPDB fungicide 分类列表。
- `label=0`：只出现在 PPDB herbicide 或 insecticide 分类列表。
- 原始 `category`、PPDB ID、来源和定义均保留。
- 旧负标签不是“对所有真菌无活性”的实验结论，而是 PPDB 类别归属。

### 3.2 新数据

- `FUNGICIDE_status=POSITIVE` 且 `contributes_to_label=1`：有明确杀菌剂
  用途/登记证据，标为 1。
- `FUNGICIDE_status=UNKNOWN`：标为 UNKNOWN，原始标签留空，绝不转 0。
- 新数据其他用途（杀虫、除草等）不推断为旧任务负样本。
- 产品级、多成分产品的用途证据不分配给单个成分；`label_evidence.csv`
  保留证据层级和 `contributes_to_label`。

### 3.3 合并后的结构标签

- 旧正 + 新正/未知 → POSITIVE。
- 旧负 + 新未知 → NEGATIVE（保留旧定义）。
- 旧负 + 新正 → CONFLICT。
- 只有新正 → POSITIVE；只有新未知 → UNKNOWN。
- CONFLICT 结构不参与普通监督训练；完整记录保留在 `data/conflicts.csv`。

冲突例子包括 Acephate、Carbaryl 相关结构和 Chlorpyrifos：旧 PPDB 将其归为
杀虫剂，新 EPA 单成分产品证据却给出 FUNGICIDE。本轮不手工改正任何来源。

新数据正标签的质量并不均匀：293 个“仅新数据正”的结构中，191 个只有
EPA PPIS 证据；只有 21 个只有 EU 证据，其余为多来源。726 个可监督正结构
中有 168 个是多成分结构。这些标签是用途/登记证据，不是实验活性，也没有
经过专家复核；它们会限制模型结论的强度。

## 4. 训练数据与划分

- 结构分组，同一规范结构只能在一个 split。
- Murcko scaffold 相同的一组结构整体分配。
- 6 个 ≥20 个结构的大 scaffold（540 个结构）强制放入训练集，避免一个
  苯环类 scaffold 占据测试集。
- 其余 scaffold 组按 seed 随机填入当前比例最低的 split，目标 70/15/15。

| split seed | train | val | test | train 正 | val 正 | test 正 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 1,454 | 312 | 312 | 476 | 122 | 128 |
| 43 | 1,454 | 312 | 312 | 487 | 123 | 116 |
| 44 | 1,454 | 312 | 312 | 500 | 113 | 113 |

主实验固定 split seed 42，训练 seed 42/43/44。测试集 312 个结构、128 个正样本；
这个规模足以计算 AUC，但仍有明显抽样波动。

## 5. 模型与训练配方

共同 backbone：冻结 ChemBERTa-zinc-base-v1 的 token states（最大 512 token，
实际最长 416，未截断）→ 128 维投影 → 4 层、4 头统一 Transformer → CLS 分类。
没有使用 ConfSeq；没有加载旧项目训练好的分类器。

PaDEL 由官方 PaDEL-Descriptor jar 计算。3,100 个结构中有 304 行至少一个
缺失值、29 行 24 列全缺失（>60 个重原子或超时）；缺失值用训练集中位数填补，
C 的辅助目标用 mask 排除缺失项。标准化只在训练集拟合。

| 组 | 输入 | 额外目标 |
| --- | --- | --- |
| A | SMILES token states | 无 |
| B | SMILES token states + 24 PaDEL descriptor tokens | 无 |
| C | 同 B | SMILES → 24 PaDEL 的 masked MSE，权重 0.2 |
| B_OLD_ONLY | 同 B | 训练集去掉“仅新数据正”的结构，用作来源消融 |

C 的辅助集包含 1,454 个训练结构，以及 scaffold 与验证/测试都不重叠的
UNKNOWN/CONFLICT 结构，共 2,345 个；不使用验证/测试结构或测试 scaffold 的
任何输入信号。

训练：AdamW，lr 3e-4，weight decay 1e-4，batch 32，warmup 10% + cosine，
最多 100 epoch，patience 18，BF16，grad clip 5.0。BCE 使用训练集
`neg/pos` 正类权重。早停只看验证 ROC-AUC；阈值由验证集最大 F1 选择；
测试集只评估一次。

## 6. 实验结果

所有数字来自固定 scaffold split 42、训练 seed 42/43/44；同一测试集 312 个
结构（128 正 / 184 负）。表中为 3 个 seed 的均值 ± 样本标准差。

### 6.1 主结果

| 实验 | 测试 ROC-AUC | 测试 PR-AUC | 测试 F1 | 测试 MCC | 验证 AUC | 训练 AUC | 训练-测试 AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A（SMILES） | 0.7593 ± 0.0173 | 0.6840 ± 0.0163 | 0.6336 ± 0.0089 | 0.3414 ± 0.0286 | 0.8507 | 0.9014 | 0.1421 |
| B（SMILES+PaDEL） | 0.7818 ± 0.0397 | 0.7028 ± 0.0557 | 0.6789 ± 0.0503 | 0.4208 ± 0.0921 | 0.8472 | 0.9634 | 0.1816 |
| C（B+辅助） | **0.7886 ± 0.0161** | **0.7106 ± 0.0275** | 0.6751 ± 0.0324 | **0.4486 ± 0.0513** | 0.8525 | 0.9780 | 0.1894 |
| B_OLD_ONLY（来源消融） | 0.7623 ± 0.0146 | 0.6533 ± 0.0262 | 0.6585 ± 0.0074 | 0.3953 ± 0.0151 | 0.8381 | 0.9516 | 0.1894 |

逐 seed 测试 AUC：

| 实验 | seed42 | seed43 | seed44 |
| --- | ---: | ---: | ---: |
| A | 0.7776 | 0.7433 | 0.7570 |
| B | 0.7362 | 0.8012 | **0.8081** |
| C | 0.7712 | 0.8029 | 0.7916 |
| B_OLD_ONLY | 0.7790 | 0.7551 | 0.7527 |

C 的平均 AUC/PR-AUC/MCC 最高，且 AUC 标准差最小；但 C 与 B 的差距只有
+0.0068，远小于 seed 间波动。B 的单次最高值（0.8081）高于 C 的任何单次，
不能只报告最好单次。

### 6.2 配对差

| 对比 | ΔAUC | ΔPR-AUC | ΔF1 | ΔMCC |
| --- | --- | --- | --- | --- |
| B − A | +0.0225 ± 0.0555 | +0.0188 ± 0.0674 | +0.0453 ± 0.0445 | +0.0795 ± 0.1101 |
| C − B | +0.0068 ± 0.0261 | +0.0078 ± 0.0354 | −0.0039 ± 0.0265 | +0.0277 ± 0.0452 |
| B − B_OLD_ONLY | +0.0195 ± 0.0543 | +0.0495 ± 0.0814 | +0.0204 ± 0.0576 | +0.0255 ± 0.0997 |
| C − A | +0.0293 ± 0.0333 | +0.0267 ± 0.0433 | +0.0414 ± 0.0241 | +0.1072 ± 0.0653 |

n=3 配对，不能据此做显著性声明。B−A 在 seed42 为负（−0.0414），
C−A 在 seed42 也为负（−0.0064）；只有 seed43/44 为正。

### 6.3 测试子集

| 实验 | 全部 AUC | 旧 PPDB 正 vs 负 | 仅新数据正 vs 负 | 多成分正 vs 负 | 单成分正 vs 负 |
| --- | --- | --- | --- | --- | --- |
| A | 0.7593 | 0.7204 ± 0.0254 | 0.8180 ± 0.0106 | 0.8989 ± 0.0126 | 0.7166 ± 0.0219 |
| B | 0.7818 | 0.7519 ± 0.0319 | 0.8269 ± 0.0545 | 0.8832 ± 0.0520 | 0.7508 ± 0.0361 |
| C | 0.7886 | 0.7518 ± 0.0203 | **0.8441 ± 0.0219** | 0.9022 ± 0.0243 | 0.7538 ± 0.0141 |
| B_OLD_ONLY | 0.7623 | 0.7548 ± 0.0132 | 0.7735 ± 0.0230 | 0.8145 ± 0.0179 | 0.7463 ± 0.0168 |

测试集共 128 个正结构：旧 PPDB 正 77 个，仅新数据正 51 个，多成分正 30 个。
多成分和“仅新数据正”子集都很小，子集 AUC 只能作为方向性证据。

### 6.4 新数据的作用

`B_OLD_ONLY` 与 B 使用完全相同模型和测试集，只把“仅新数据正”的 195 个
训练结构去掉（训练结构 1,454 → 1,259）：

- 总体 AUC：B 0.7818 vs B_OLD 0.7623，差 +0.0195 ± 0.0543。
- 旧 PPDB 正子集：B 0.7519 vs B_OLD 0.7548，基本不变（−0.0029）。
- 仅新数据正子集：B 0.8269 vs B_OLD 0.7735，差 +0.0534。

因此，新数据的主要作用是对“新化学空间中的正结构”提供训练信号；它没有
明显改善旧 PPDB 正结构的排序。总体增益被 seed 间波动掩盖，不能称为稳定提升。
另外，新数据没有负标签，负样本仍全部来自旧 PPDB；这使“新数据作用”和
“来源差异”无法完全分开。

### 6.5 PaDEL 是否被使用

对 B 和 C 做测试期 PaDEL 置换（只打乱测试描述符，SMILES、模型和标签不变）：

| 模型 | 种子数 | 行置换平均 ΔAUC | 列置换平均 ΔAUC |
| --- | ---: | ---: | ---: |
| B | 3 | 0.00080 | 0.00067 |
| C | 2 | 0.00037 | 0.00029 |

AUC 几乎不变（<0.002），说明 B/C 的 CLS 分类路径基本没有利用 24 个 PaDEL。
B 相对 A 的平均 +0.0225 不能归因于描述符信息；C 相对 B 的 +0.0068 也不能
归因于描述符被主路径使用。它们更可能是额外 token/辅助损失带来的优化或
正则化差异。C 的辅助 MSE 最终约 0.92–1.00（标准化目标），也没有显示强拟合。

### 6.6 阈值与混淆矩阵（seed42）

| 实验 | 阈值 | TN | FP | FN | TP | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 0.8176 | 135 | 49 | 47 | 81 | 0.623 | 0.633 | 0.628 |
| B | 0.7295 | 117 | 67 | 40 | 88 | 0.568 | 0.688 | 0.622 |
| C | 0.2870 | 138 | 46 | 45 | 83 | 0.643 | 0.648 | 0.646 |
| B_OLD_ONLY | 0.0289 | 122 | 62 | 33 | 95 | 0.605 | 0.742 | 0.667 |

阈值只由验证集最大 F1 选择；测试集没有参与阈值或模型选择。

### 6.7 训练成本

| 实验 | 参数量 | 最佳 epoch 均值 | 总 epoch 均值 | 秒/epoch | 总训练秒 | 峰值显存 MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 703,489 | 8.7 | 26.7 | 1.57 | 42.8 | 294 |
| B | 715,009 | 19.7 | 37.7 | 1.29 | 49.6 | 311 |
| C | 725,081 | 24.0 | 42.0 | 1.57 | 67.3 | 370 |
| B_OLD_ONLY | 715,009 | 14.7 | 32.7 | 1.16 | 38.7 | 189 |

GPU：RTX 4060 Laptop 8 GB；BF16。所有 run 均未 OOM。C 比 B 多约 36% 训练
时间、约 19% 峰值显存，换来 +0.0068 AUC（不稳健）。

### 6.8 图

- `figures/auc_pr_summary.png`：A/B/C 的测试 AUC/PR-AUC 和逐 seed 点。
- `figures/source_subset_auc.png`：旧正 vs 仅新正子集。
- `figures/padel_permutation.png`：B 的 PaDEL 置换。

## 7. 结论

### 7.1 合并数据是否形成了有意义的模型？

形成了一个**有内部区分信号、但不能称为可靠杀菌剂活性预测器**的模型：
scaffold 测试 AUC 约 0.76–0.79，PR-AUC 约 0.68–0.71，明显高于随机；
但测试集只有 128 个正结构，标签是用途/登记证据而非实验活性，且没有外部
活性验证。训练-测试 AUC 差 0.14–0.19，过拟合明显。

### 7.2 新数据是否帮助？

**有小幅、方向性但不稳健的帮助。** B 比只用旧标签的 B_OLD_ONLY 高
+0.0195 AUC，主要来自“仅新数据正”子集（+0.0534）；旧正子集没有改善。
新数据没有负标签，且 293 个仅新正结构中有 191 个只有 EPA 单成分产品证据，
所以这部分收益也可能是学习到来源/产品标签噪声，而不是杀菌机制。

### 7.3 24 个 PaDEL 是否有效？

**没有证据表明有效。** B−A 平均 +0.0225，但逐 seed 一正一负一正，
且测试期置换只下降 0.0008（行）/0.0007（列）。C 同样只有 0.0004/0.0003。
因此不能把 B/C 的任何提升归因于 PaDEL；B 的额外描述符 token 基本被忽略。

### 7.4 辅助学习是否有效？

**不能确认。** C 的平均 AUC/PR-AUC/MCC 最高，AUC 标准差最小，但 C−B 仅
+0.0068 ± 0.0261，且 PaDEL 置换显示主路径没有使用描述符。辅助损失约 1.0，
没有强拟合。它可能作为正则化略微稳定训练，但不是已验证的辅助学习收益。

### 7.5 哪个方案最值得继续？

- **部署/基线：A（SMILES）**。最简单，平均 AUC 0.759，与 C 的 0.789 差距
  小于 seed 波动；没有证据表明额外 PaDEL/辅助头值得成本。
- **研究继续：C**。如果继续，应把它当作“带正则化的 SMILES 模型”而不是
  “PaDEL 融合模型”；先修复标签和负样本，再验证辅助任务是否真的传递信息。
- **不建议把 B 作为 PaDEL 方案继续**：置换检验已经否定了描述符被使用。
- 更重要的是补数据质量：明确负证据、复核 EPA 单成分产品标签、增加独立
  外部测试集；单纯增加同源记录不会自动解决这些问题。

## 8. 失败项与限制

1. **PaDEL 融合失败**：B/C 的测试期置换几乎不改变 AUC；24 个描述符没有被
   主分类器有效利用。
2. **辅助任务弱**：C 的辅助 MSE 没有明显下降，C−B 在噪声范围内。
3. **新数据增益不稳健**：B−B_OLD 的均值 +0.0195，但 seed42 为负；总体
   被 seed 波动掩盖。
4. **标签不是实验活性**：新数据 FUNGICIDE 是登记/产品用途证据；不能把模型
   输出解释为“对真菌有活性”的概率。
5. **新数据标签质量不均**：293 个仅新正结构中 191 个只有 EPA PPIS 证据；
   42 个结构与旧负标签冲突（110 条记录）被整体排除。没有手工改正来源。
6. **负样本单一**：1,352 个负结构全部来自旧 PPDB herbicide/insecticide 类别；
   新数据没有负标签。正负样本的来源不平衡会混淆“杀菌剂信号”和“数据来源”。
7. **小测试集**：测试 312 个结构、128 个正；子集更小（新仅正 51、多成分正 30），
   子集 AUC 不稳定。
8. **单一 scaffold split**：主结果只使用 split seed 42；虽然训练了 3 个
   model seed，但没有覆盖 scaffold 划分不确定性。43/44 划分已生成但未用于
   主表，避免事后挑选。
9. **来源消融的早停不完全对等**：B_OLD_ONLY 仍在完整验证集（含仅新数据正
   结构）上早停，B 与 B_OLD 的差异因此只能作为方向性来源证据，不是严格
   控制实验。
10. **多成分结构**：260 个可监督多成分结构（168 正）被保留；盐/制剂结构可能
   不是单一活性成分，测试子集只有 30 个多成分正，结论不稳。
11. **过拟合**：训练-测试 AUC 差 0.14–0.19；早停后仍有明显记忆。
12. **PaDEL 缺失**：304 行至少一个缺失值，29 行全部缺失；用训练中位数填补，
    辅助 loss 用 mask 排除。缺失值可能影响 B/C。
13. **无外部验证**：没有新的独立实验活性数据；不能声称泛化到真实新化学空间。
14. **统计功效**：3 个 seed、n=3 配对差，未做显著性检验；不要把均值差异
    当作已证明的改进。

## 9. 复现与产物

```bash
cd /d/works/yuce
./.venv/Scripts/python.exe experiments/merged_fungicide/scripts/build_dataset.py
./.venv/Scripts/python.exe experiments/merged_fungicide/scripts/make_splits.py
./.venv/Scripts/python.exe experiments/merged_fungicide/scripts/prepare_features.py
./.venv/Scripts/python.exe experiments/merged_fungicide/scripts/train_models.py \
  --experiments A,B,C --split-seeds 42 --seeds 42,43,44
./.venv/Scripts/python.exe experiments/merged_fungicide/scripts/train_models.py \
  --experiments B_OLD_ONLY --split-seeds 42 --seeds 42,43,44
./.venv/Scripts/python.exe experiments/merged_fungicide/scripts/analyze_results.py
./.venv/Scripts/python.exe experiments/merged_fungicide/scripts/padel_permutation.py \
  --experiment B --split-seed 42 --seed 42
./.venv/Scripts/python.exe experiments/merged_fungicide/scripts/make_figures.py
./.venv/Scripts/python.exe experiments/merged_fungicide/scripts/make_manifest.py
```

| 产物 | 说明 |
| --- | --- |
| `reports/data_audit.md` | 数据/标签/冲突/划分审计 |
| `data/merged_fungicide_records.csv` | 6,592 条记录 |
| `data/structures.csv` | 3,100 个唯一结构 |
| `data/label_evidence.csv` | 24,668 条证据 |
| `data/unknown_records.csv` | 2,618 条 |
| `data/conflicts.csv` | 110 条冲突记录 |
| `data/positive_without_structure.csv` | 375 条 |
| `splits/` | 3 个 scaffold split + 摘要 |
| `features/feature_manifest.json` | ChemBERTa/PaDEL 特征清单 |
| `configs/A.yaml`、`B.yaml`、`C.yaml`、`B_OLD_ONLY.yaml` | 配置 |
| `checkpoints/` | 12 个 run 的权重、history、config、预测 |
| `metrics/tables.md` | 全部结果表 |
| `metrics/padel_permutation_summary.json` | 置换汇总 |
| `metrics/isolation_audit.json` | 结构/辅助集隔离检查 |
| `metrics/summary.json` | 机器可读均值 |
| `figures/` | 3 张图 |
| `logs/code_manifest.json`、`data/data_manifest.json` | 哈希清单 |

## 13. APPf 风格原理样机（后续子实验）

在 `experiments/merged_fungicide/appf_like/` 中新增了一个独立的 APPf 风格
CatBoost 原理样机：

- 严格 24 PaDEL + CatBoost + 25 个平衡子集：测试 AUC 0.737 ± 0.062；
- 扩展 24 PaDEL + RDKit + Morgan/MACCS：测试 AUC 0.872 ± 0.018；
- split 42 扩展模型：AUC 0.853（95% CI 0.809–0.892），Top-10% precision
  0.903；
- 旧数据训练 → 仅新数据正：AUC 0.821 ± 0.025；50 次标签置换 p = 0.020；
- 提供 Streamlit demo、model card、适用域和 SHAP 解释。

边界仍然是“杀菌剂用途/登记证据的 like 模型”，不是实测活性。完整报告见
`appf_like/reports/APPF_LIKE_REPORT.md`，给老师的一页说明见
`appf_like/reports/DEMO_ONE_PAGER.md`。
