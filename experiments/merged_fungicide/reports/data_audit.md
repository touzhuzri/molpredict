# 合并杀菌剂数据集：数据与标签审计

生成目录：`D:\works\yuce\experiments\merged_fungicide`
生成脚本：`scripts/build_dataset.py`、`scripts/make_splits.py`

## 0. 实际路径

- 旧项目实际根目录：`D:\works\yuce`。
- 任务描述中的 `D:\works\yuce\molpredict` **不存在**；旧项目的 `data/`、`src/`、`experiments/` 位于 `D:\works\yuce` 根目录。本轮按实际根目录读取，未寻找其他项目替代。
- 新数据：`D:\works\yuce\shujv\pesticide-data\data\processed\d2_1_training`，只读。
- 旧数据：`D:\works\yuce\data\training\ppdb_labeled.csv`，只读。
- 所有新产物只写入 `experiments/merged_fungicide/`。

旧项目同时存在 `confseq.csv`（1,823 行）和 `ppdb_labeled.csv`（1,827 行）。`confseq.csv` 是去掉 4 条 ConfSeq 转换失败记录后的派生文件；本轮不使用 ConfSeq，因此采用完整的 `ppdb_labeled.csv`，保留全部 1,827 条旧记录。

## 1. 两个来源的原始记录

### 1.1 旧数据 PPDB

- 原始记录：1,827。
- 阳性 `label=1`：433。
- 阴性 `label=0`：1,394。
- 唯一 InChIKey / 唯一 SMILES：1,827 / 1,827。
- 结构：全部可解析。
- 标签定义（保留原义）：
  - 1：只出现在 PPDB fungicide 分类列表；
  - 0：只出现在 PPDB herbicide 或 insecticide 分类列表；
  - 同时出现在杀菌剂与其他类别的 mixed 记录已在旧项目中排除。
- 这是 **PPDB 分类标签**，不是实验测定的杀菌活性值。

### 1.2 新数据 D2.1

- 新项目上游来源记录：约 133,404 条；产品/成分记录不是分子记录，不能把
  产品数当分子数。
- 去重后的候选物质：4,765。本节以下计数以候选物质为记录单位。
- 可解析结构：2,924（61.36%）。
- 无结构：1,656。
- AMBIGUOUS：185。
- VERIFIED：1；PROVISIONAL：2,923。
- SINGLE_COMPONENT：2,147；MULTI_COMPONENT：777。
- FUNGICIDE 标签：
  - POSITIVE：1,008 条记录；
  - UNKNOWN：3,757 条记录；
  - 明确阴性：0；
  - 标签冲突：0（新数据内部）。
- 新数据标签是登记/产品用途证据，不是实验活性。`label_evidence.csv` 中 `contributes_to_label=1` 的 FUNGICIDE 证据才进入候选标签；产品级、多成分归因不进入候选标签。
- 新数据 `training_eligible` 1,889（MAIN 670 / EXTENDED 1,219）；本轮不把它作为数据上限，而是从 4,765 个候选重新构建。

## 2. 合并后的规模

| 指标 | 数量 |
| --- | ---: |
| 合并记录总数 | 6,592 |
| 旧记录 | 1,827 |
| 新记录 | 4,765 |
| 可解析结构记录 | 4,751 |
| 唯一规范化结构 | 3,100 |
| 可解析但重复的结构记录 | 1,651 |
| 旧/新都出现的结构 | 1,015 |
| 仅旧数据的结构 | 812 |
| 仅新数据的结构 | 1,273 |
| 结构缺失记录 | 1,841 |
| 多成分记录 | 915 |

说明：6,592 条“记录”、3,100 个“唯一结构”、2,078 个“可监督结构”不是同一层级。记录数保留来源和证据；结构数用于防止同一分子跨集合；可监督数才进入二分类损失。

## 3. 统一标签状态

每个结构只取以下状态之一：

- `POSITIVE`：至少一个来源有明确杀菌剂用途/分类证据；
- `NEGATIVE`：旧 PPDB 明确只属于除草剂/杀虫剂类别，且没有新数据杀菌剂正证据；
- `UNKNOWN`：只有新数据的 UNKNOWN，或没有足够证据；
- `CONFLICT`：同一结构同时有正、负证据。

### 3.1 结构级标签

| 状态 | 结构数 |
| --- | ---: |
| POSITIVE | 726 |
| NEGATIVE | 1,352 |
| UNKNOWN | 980 |
| CONFLICT | 42 |
| 合计 | 3,100 |

### 3.2 记录级标签（原始）

| 来源 | POSITIVE | NEGATIVE | UNKNOWN |
| --- | ---: | ---: | ---: |
| 旧 PPDB | 433 | 1,394 | 0 |
| 新 D2.1 | 1,008 | 0 | 3,757 |

新数据的 UNKNOWN 没有转成 0。旧数据的负标签保留原 PPDB 定义和来源。

### 3.3 正样本来源贡献（可监督结构）

| 正样本来源 | 结构数 |
| --- | ---: |
| 仅旧 PPDB 正标签 | 224 |
| 旧、新都有正证据 | 209 |
| 仅新数据正证据 | 293 |
| 合计 | 726 |

- 旧 PPDB 正结构共 433 个：224 + 209。
- 新数据正结构共 544 个：209 + 293 + 42 个冲突结构中的新正证据（冲突结构整体不训练）。
- 新数据补充了 293 个旧数据没有的正结构，但其中有相当一部分只有 EPA PPIS 单成分产品证据；标签质量见第 8 节。

### 3.4 负样本来源

- 可靠负样本：1,352 个结构，全部来自旧 PPDB 的 herbicide/insecticide 类别。
- 新数据没有明确负标签；新数据“没有杀菌剂记录”没有被当作负样本。
- 有 734 个负结构同时有新的 UNKNOWN 记录，仍保留旧负标签；新 UNKNOWN 不构成反驳。
- 因此，监督信号是 **旧负样本 + 旧/新正样本**，负样本来源单一，这是本轮的重要混杂因素。

## 4. 冲突（CONFLICT）

- 冲突结构：42 个。
- 冲突记录：110 条。
  - 旧 PPDB NEGATIVE：42 条；
  - 新 D2.1 POSITIVE：44 条；
  - 新 D2.1 UNKNOWN：24 条。
- 典型冲突包括 Acephate、Carbaryl 相关结构和 Chlorpyrifos：旧 PPDB 将其归为杀虫剂，新 EPA 产品证据却给出 FUNGICIDE。这里不判断谁对谁错，也不静默覆盖；整个结构从普通监督训练中排除。
- 完整清单：`data/conflicts.csv`。

## 5. UNKNOWN 与不可用正样本

- 新原始 UNKNOWN 记录：3,757。
- 其中：
  - 1,466 条无结构；
  - 958 条落在旧 PPDB 已知负结构上（结构级仍为 NEGATIVE，UNKNOWN 没有覆盖旧标签）；
  - 1,152 条落在结构级 UNKNOWN 的结构上；
  - 157 条落在其他记录提供正证据的结构上（结构级仍为 POSITIVE）。
- `unknown_records.csv` 收录结构级 UNKNOWN 的记录，以及无结构的 UNKNOWN 记录，共 2,618 条。
- 有 375 条新数据 FUNGICIDE POSITIVE 记录没有可解析结构；它们保留在 `positive_without_structure.csv`，不能进入结构模型。

## 6. 可监督训练集

- 可监督结构：**2,078**。
  - 正：726（34.9%）
  - 负：1,352（65.1%）
- 多成分结构：260，其中正 168；单成分正 558。
- 结构级 scaffold 组：1,019（其中无环结构按精确结构单独成组）。
- 全部 3,100 个可解析结构都保留在 `structures.csv` 和特征缓存中；UNKNOWN 不进入监督损失。C 组的无标签辅助任务只使用“训练结构 + scaffold 与验证/测试均不重叠的 UNKNOWN/CONFLICT 结构”，避免把验证/测试结构或测试 scaffold 的输入带入任何训练信号。

## 7. 划分

采用规范化结构分组的 scaffold split：

1. 同一结构只能在一个 split；
2. Murcko scaffold 相同的一组结构整体分配；
3. 含 20 个及以上结构的 scaffold（6 组、540 个结构）强制放入训练集，避免一个苯环类大 scaffold 占据测试集；
4. 其余 scaffold 组按 seed 随机顺序，填入当前比例最低的 split，目标 70/15/15。

| split seed | train | val | test | train 正 | val 正 | test 正 | 测试 scaffold 数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 1,454 | 312 | 312 | 476 | 122 | 128 | 195 |
| 43 | 1,454 | 312 | 312 | 487 | 123 | 116 | 200 |
| 44 | 1,454 | 312 | 312 | 500 | 113 | 113 | 203 |

- 三个 split 的所有 scaffold 交集均为 0。
- 测试集正样本 113–128，规模可计算 AUC/PR-AUC，但仍属于小测试集，指标有抽样波动。
- 测试集正比例（36%–41%）高于训练集（33%–34%），是因为 scaffold 组整体分配，不能通过重新抽样“调平”；这没有使用测试标签做任何模型选择。
- 所有模型共用同一 split；测试集只在早停和阈值确定后评估一次。

## 7.1 特征

- ChemBERTa：3,100 条结构全部编码；最大输入长度 512，实际最长 416，
  截断 0 条。
- PaDEL：3,100 行 × 24 列；304 行至少有一个缺失值，29 行全部 24 列缺失
  （分子 >60 个重原子或 PaDEL 超时）。缺失值只在训练集拟合中位数填补；
  C 的辅助损失用 mask 排除缺失目标。
- 所有特征缓存按 `structures.csv` 的 `structure_index` 对齐。

## 8. 标签质量与限制

1. **新正标签不是实验活性。** 它表示登记/产品用途证据；不能解释为“对某种真菌有活性”。
2. **EPA 单成分产品证据可能很噪。** 例如 Chlorpyrifos、Acephate 被新 EPA 产品标签归为 FUNGICIDE，而旧 PPDB 将其归为杀虫剂。本轮将这些结构标为 CONFLICT 并排除，不手工改正任何来源。
3. **新数据没有负标签。** 负样本全部来自旧 PPDB，模型可能学到“来源/化学空间”差异，而不是杀菌剂机制。
4. **结构身份是规范 SMILES 精确匹配。** 盐、母体、立体异构体可能被当作不同结构；多成分结构保留，但其标签可能来自盐/制剂而非单一活性成分。
5. **MAIN/EXTENDED 不是 VERIFIED。** 新数据 2,923/2,924 个结构为 PROVISIONAL；本轮没有把它当作已验证结构。
6. **记录重复不等于独立证据。** 同一结构的多条产品记录可能来自同一登记来源；训练时按结构只取一个标签，避免重复记录放大权重。
7. **同 scaffold 允许跨新旧数据。** 本轮按任务要求不再因为相似/同骨架而删除；只在划分时保证同一结构不跨集合。

## 9. 审计产物

- `data/merged_fungicide_records.csv`：6,592 条记录，保留原始 ID、来源、原始标签、规范结构、结构标签和排除原因。
- `data/label_evidence.csv`：24,668 条标签证据（旧 1,827 + 新 FUNGICIDE 证据 22,841）。
- `data/unknown_records.csv`：2,618 条。
- `data/conflicts.csv`：110 条冲突记录。
- `data/positive_without_structure.csv`：375 条。
- `data/structures.csv`：3,100 个唯一结构及 scaffold、来源计数、结构标签。
- `data/dataset_statistics.json`：机器可读统计。
- `splits/split_seed{42,43,44}.json`、`splits/split_summary.json`。
