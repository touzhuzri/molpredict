# INITIAL_AUDIT — 旧项目与新数据检查

日期：2026-11-04
工作目录：`D:\works\yuce`
旧项目：`D:\works\yuce`（molpredict）
新数据（只读）：`D:\works\yuce\shujv\pesticide-data\data\processed\d2_1_training`

本文件是实验前的审计记录。新项目只被读取，未修改任何文件；V2/V3/V3.1 的历史产物未被覆盖。

## 1. 旧项目数据

- 主数据：`data/training/confseq.csv`
- 样本：**1,823** 个分子；**431** 阳性 / **1,392** 阴性。
- 标签来源：PPDB 农药分类。阳性 = PPDB fungicide 列表；阴性 = PPDB
  herbicide 或 insecticide 列表。**不是实验抗真菌活性**，阴性也不等于实验无活性。
- 旧项目已转换 ConfSeq 的分子是 1,823/1,827（4 个转换失败未进入本表）。
- 旧数据还包含 SMILES、ConfSeq 字符串和 24 个 PaDEL 描述符。

### 1.1 旧划分

旧项目使用 V2 的 Murcko scaffold 划分，文件：

- `experiments/v2/splits/split_seed42.json`
- `experiments/v2/splits/split_seed43.json`
- `experiments/v2/splits/split_seed44.json`

实际索引数量：

| seed | train | val | test | test 阳性 |
|---|---:|---:|---:|---:|
| 42 | 1,332 | 168 | 323 | 80 |
| 43 | 1,320 | 167 | 336 | 82 |
| 44 | 1,295 | 208 | 320 | 83 |

索引互不重叠；`experiments/v3/leakage_audit.json` 记录 InChIKey、canonical SMILES、
Murcko scaffold 的 train/val/test 重叠均为 0。预训练语料重叠状态为 `unknown`。

## 2. 旧模型与输入

### 2.1 V3.1 统一模型

- 代码：`src/molpredict/v3_model.py`、`src/molpredict/v3_1_train.py`
- 结构：`[CLS] + SMILES token states + ConfSeq token states + 24 descriptor tokens`
  进入同一个 Transformer；CLS 后接分类头。
- V3.1 正式基线：`d_model=128`、4 层、4 头、FFN 256、dropout 0.3、
  `pooling=cls`；融合网络 789,121 参数。
- V3.1 `full` / `base` 确实使用三个输入：SMILES、ConfSeq、24 PaDEL
  （见 `experiments/v3_1/group1/runs/V3_1_base_seed42/config.json` 的
  `modalities` 和 `attention.json`）。
- 24 个 descriptor 是独立 token：每项 = MLP(标准化标量) + 身份 embedding；
  全部 24 个 token 有效，顺序固定为 `feature_order.json` 中的 `padel24`。
- 训练配方：AdamW，融合头 lr=3e-4，编码器 lr=2e-5（部分解冻时），
  `BCEWithLogitsLoss(pos_weight=neg/pos)`，batch 16×2 梯度累积，
  warmup 10% + cosine 到 0.1×，最多 100 epoch，验证 ROC-AUC 早停 patience=18，
  阈值由验证集最大 F1 选择，测试集只评估一次。

### 2.2 24 项 PaDEL 特征

名称与顺序（`src/molpredict/padel.py`）：

```text
nBr, AATS7m, ATSC0c, AATSC5m, AATSC6m, VR1_DzZ, nHdNH, nHsSH, nHsNH2,
nHsssNHp, nHCsatu, nssssNp, naaO, nssssSi, nddssS, SsBr, minHBint8,
minHBint9, n7Ring, nG12Ring, nF6Ring, nF7Ring, nF8Ring, n6HeteroRing
```

- 由 padelpy 0.1.17 内置 PaDEL-Descriptor jar 计算，只启用 2D 相关组；
  不用 RDKit 同名/相似描述符替代。
- 旧缓存：`experiments/v2/cache/padel24.csv`；顺序：`padel24_order.json`。
- 标准化：`NumericStandardizer` 只在训练集拟合；缺失值用训练集中位数填补，
  再减均值除标准差。24 列都有缺失，列缺失数 11–58。

### 2.3 历史结果（只作参考，不写入新实验）

| 模型 | 测试 AUC | 说明 |
|---|---:|---|
| CatBoost A | 0.8578 ± 0.0292 | 旧最佳基线 |
| V2 F | 0.7602 ± 0.0062 | 池化融合 |
| V2 F-ft | 0.8159 ± 0.0197 | 部分微调 |
| V3.1 `base`（128/4/4，冻结） | 0.7718 ± 0.0312 | 三 seed |
| V3.1 `ft_both` | 0.8342 ± 0.0307 | 三 seed |

### 2.4 已知问题

- PPDB 标签是分类，不是实验活性；431 阳性，验证集 167–208 条。
- 三套 scaffold 测试集彼此有重叠；mean±std 不是三次独立复现。
- 早停按验证 AUC，验证损失后期常升到 1.5–3.0，checkpoint 校准较差。
- 预训练语料（ChemBERTa/ZINC、ConfSeq）重叠为 `unknown`。
- 外部 21 个已知分子上 V3.1 AUC 0.462–0.673，低于 CatBoost 0.750；
  真正 scaffold-disjoint 子集上全部接近/低于随机。

## 3. 新数据审计

路径：`shujv/pesticide-data/data/processed/d2_1_training`（只读）。

文件：`molecules.csv`、`labels.csv`、`label_masks.csv`、`label_evidence.csv`、
`molecules_with_labels.csv`、`training_eligible.csv`、`structure_audit.csv`、
`dataset_statistics.json`、`splits/`（`random_*`、`scaffold_*`）。

### 3.1 实际数量

| 项目 | 实际值 |
|---|---:|
| 全部候选 | 4,765 |
| 可解析结构 | 2,924 |
| 结构状态 VERIFIED | 1 |
| PROVISIONAL | 2,923 |
| AMBIGUOUS | 185 |
| NO_STRUCTURE | 1,656 |
| 有结构且至少一个已知正标签 | 1,889 |
| MAIN（training_eligible） | 670 |
| EXTENDED（training_eligible） | 1,219 |
| training_eligible 唯一 InChIKey | 1,529 |
| 其中 SINGLE_COMPONENT | 1,463 |
| 其中 MULTI_COMPONENT | 426 |

- MAIN 和 EXTENDED 的 `substance_id` 不相交。
- **MAIN 不是 VERIFIED-only**：MAIN 中绝大多数结构为 PROVISIONAL；
  VERIFIED 只有 1 个。
- 结构冲突、重复结构、多组分：`structure_audit.csv` 和
  `structure_duplicate_group_size` 可查；本实验会排除 MULTI_COMPONENT、
  AMBIGUOUS、NO_STRUCTURE 和不可解析结构。

### 3.2 用途标签

`labels.csv` 的 10 个用途列只有：

- `POSITIVE`：有登记/用途证据；
- `UNKNOWN`：无已知证据；
- **明确负样本 0，冲突 0**。

`label_masks.csv` 中 mask=1 仅表示已知正标签；mask=0 是未知。**UNKNOWN 不是 0**。
因此不能用普通二分类辅助任务，也不能把新数据 FUNGICIDE 当作旧任务的活性正样本。

### 3.3 与旧数据的重叠

按 InChIKey 比对：

- training_eligible 中 **1,130 行 / 900 个唯一 InChIKey** 已在旧项目 1,823 个
  分子中；不能作为“新结构”参与辅助训练。
- 排除旧分子、MULTI_COMPONENT、不可解析结构并按 InChIKey 去重后：
  - 唯一新结构 **390** 个（MAIN 79，EXTENDED 311）；
  - 其中 **191** 个与旧 val/test 的 Murcko scaffold 重叠。
- 采用更严格的隔离（再排除旧 val/test scaffold 重叠）后：
  - MAIN **45**、MAIN+EXTENDED **199** 个可用于辅助学习。

本实验的 `D_NEW_AUX` 主分析使用严格隔离的 **45 / 199** 两个规模；390 的
exact-only 版本仅作为敏感性检查（如运行）。所有排除数量和原因记录在
`data_audit/new_aux_audit.json`。

### 3.4 新数据划分

新项目自带 `splits/random_*` 和 `splits/scaffold_*`，但本实验**不使用**这些
划分作为主任务划分；主任务固定使用旧项目的 seed 42/43/44。新数据只进入辅助
任务，且排除旧 val/test 分子及其 scaffold 重叠。

## 4. 对实验设计的直接影响

1. **A/B/C/E 的 NEW_AUX 条件无法合理实施**：它们没有辅助任务，把新数据加入
   主任务会改变主任务数据或标签语义。按任务要求记录为 `N/A`，不伪造替代实验。
2. **D 是唯一可合理使用新数据的组**：主任务仍用旧 PPDB 标签；辅助任务为
   “从 SMILES 表示预测 24 项 PaDEL 描述符”。新数据只提供结构和计算出的
   PaDEL，不提供旧任务标签。
3. **用途标签辅助任务不实施**：新数据只有正标签、没有明确负样本；把 UNKNOWN
   当负样本违反数据规则。强行训练只会在全部类别上输出 1。这个限制会写入最终报告。
4. **24 项 PaDEL 的有效性**通过 A/B/C 对照 + C 的测试期置换检验判断，不能只看
   分支存在或梯度非零。
5. **网络深度**只改变 C 的层数（4 → 8），其他结构/训练预算保持一致。

## 5. 待办与风险

- 新数据 MAIN/EXTENDED 的可用辅助结构很少（45/199），辅助学习的效果可能很小；
  结果必须如实报告，不能因效果弱而修改隔离规则。
- 新数据结构多为 PROVISIONAL，PaDEL 依赖 PubChem SMILES 的正确性；会记录
  无法计算/缺失的描述符。
- 旧任务标签不是实验活性；新数据用途标签也不是实验活性。最终报告不能把
  任何一项描述成抗真菌活性。
- 训练/测试仍使用旧 scaffold 划分；不会为了让某组好看而改变划分。
