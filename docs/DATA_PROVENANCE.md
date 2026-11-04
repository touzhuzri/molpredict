# 原始数据来源、许可与仓库边界

日期：2026-11-04  
适用目录：`D:\works\yuce`

## 0. 结论先行

**GitHub 仓库只提交代码、配置、报告、聚合指标和图表，不提交原始数据、
派生结构表、特征缓存、模型权重和用户上传的 Excel。**

原因不是“懒得传”，而是：

- PPDB 的许可状态为 `RESTRICTED`，不允许再分发或构建可替代数据库；
- 新 D2.1 数据混合了 `LICENSE_UNKNOWN`、`CC_BY` 等来源，不能整体再分发；
- 用户上传的 APPf 任务列表属于用户侧数据；
- 模型权重和特征缓存是这些受限数据的派生结果，许可状态不清晰。

本文件给出本地路径、行数、SHA256 和重建方法。拿到原始数据的人可以在本地
逐字节校验并复现；没有原始数据的人只能复现代码和聚合结果，不能拿到分子表。

## 1. 原始数据清单

### 1.1 旧项目 PPDB 数据

| 项目 | 内容 |
| --- | --- |
| 文件 | `data/training/ppdb_labeled.csv` |
| 行数 | 1,827 |
| 标签 | 阳性 433（PPDB fungicide），阴性 1,394（只出现在 herbicide/insecticide） |
| 结构 | 1,827 条可解析记录 |
| 来源 | Pesticide Properties Database (PPDB), University of Hertfordshire |
| URL | <https://sitem.herts.ac.uk/aeru/ppdb/en/> |
| 抓取脚本 | `scripts/fetch_ppdb.py` |
| 抓取日期 | 2026-11-03 |
| SHA256 | `986a932022a25df495859195599656f75502101ace8e6d5c2b4e0f89397a59a4` |
| 许可 | `RESTRICTED`；只做非实质性抽取/研究使用，不再分发 |
| 是否提交 GitHub | **否** |

说明：`label=0` 是“只出现在 PPDB 除草剂/杀虫剂列表”，不是“实验证明对真菌
无活性”。`label=1` 是 PPDB 分类归属，不是实验 MIC/EC50。

### 1.2 旧项目 ConfSeq 派生表

| 项目 | 内容 |
| --- | --- |
| 文件 | `data/training/confseq.csv` |
| 行数 | 1,823（4 条 ConfSeq 转换失败） |
| 来源 | 由 `ppdb_labeled.csv` 经官方 ConfSeq demo 生成 |
| SHA256 | `1ee20875e45954719da06244167bc8ff09e895f5417fabe7b1918528e3bbca76` |
| 是否提交 GitHub | **否** |

该文件不是本项目新合并模型的训练输入；保留用于旧版复现。

### 1.3 新项目 D2.1

目录：`shujv/pesticide-data/data/processed/d2_1_training/`

| 文件 | 行数/规模 | SHA256 | 说明 |
| --- | ---: | --- | --- |
| `molecules.csv` | 4,765 个候选 | `7eaecefa137bc29872a75793356172d5b826c79f9654b466cb00acc666873e56` | 候选物质、结构、标识 |
| `labels.csv` | 4,765 行 | `9734687b58957853ece3d81636f3cde6475a6c508b0e1cb6639ffd47e6c50e12` | 各用途的 POSITIVE/UNKNOWN |
| `label_evidence.csv` | 22,841 条 | `ebb069ac0d9330d6821d7950e50422f2e3d205ba0c9ce9e8dbff0f80995b8d63` | 来源级标签证据 |
| `training_eligible.csv` | 1,889 | `c3be55671e21b3562aad994a915e3d3f3308ea0f0969841396e7bb77e55affe2` | MAIN 670 / EXTENDED 1,219 |

关键统计：

- 可解析结构 2,924；
- FUNGICIDE POSITIVE 记录 1,008；
- FUNGICIDE UNKNOWN 记录 3,757；
- 明确阴性：**0**。

来源与许可（详见 `shujv/pesticide-data/docs/LICENSE_AUDIT.md`）：

- EPA PPIS/PPLS：`LICENSE_UNKNOWN`，不批量镜像、不再分发；
- Canada PPID：`LICENSE_UNKNOWN`，报告为 OGL-Canada 但未直接核验；
- EU Pesticides DB：`CC_BY_4_0`；
- Brazil Agrofit CSV：`CC_BY`；
- PubChem：`LICENSE_UNKNOWN`；
- 新数据项目明确要求：受限来源样本不得对外发布，EPA/Canada/PubChem 输出
  保持 `LICENSE_UNKNOWN`，不要再分发。

因此 D2.1 整体**不能作为可再分发数据集提交到 GitHub**。

### 1.4 用户上传的 APPf 任务列表

| 项目 | 内容 |
| --- | --- |
| 文件 | `新建 Microsoft Excel 工作表.xlsx` |
| 行数 | 677 |
| 唯一原始 SMILES | 280 |
| 缺失 SMILES | 79 |
| 可解析行 | 590 |
| 唯一有效结构 | 263 |
| 来源 | 中国农业大学 APPf 预测服务的任务列表导出 |
| SHA256 | `eb64a90d370a4c30a0b73c6e9d72fa10727d1674b4431ad0d2eaa4da3ea326d7` |
| 是否提交 GitHub | **否** |

该文件只包含任务列表，不包含 APPf 的预测结果。本项目的批量预测输出在：

```text
experiments/merged_fungicide/appf_like/predictions/excel_runs/
```

该目录包含用户 SMILES 和逐条预测，属于用户侧数据，不提交。

### 1.5 APPf 论文与补充材料

- 论文：Jin et al., *Advanced Agrochem* 4 (2025) 373–381,
  DOI `10.1016/j.aac.2025.10.005`。
- 补充材料只包含模型性能表、24 个描述符定义和 CatBoost 参数；
- 论文的 489 个杀菌剂 / 6,409 个负样本原始分子列表未公开，需向通讯作者
  申请；
- 本仓库**不包含**论文 PDF、补充 DOCX 或任何论文原始分子表。

## 2. 合并后的派生数据

由 `experiments/merged_fungicide/scripts/build_dataset.py` 生成，位于
`experiments/merged_fungicide/data/`：

| 文件 | 行数/规模 | SHA256 | 说明 |
| --- | ---: | --- | --- |
| `merged_fungicide_records.csv` | 6,592 条 | `1d3d53106d9382079e359c4509f6f5f97c66cad4b6e03180c8353283ff20e21b` | 保留旧/新 ID、来源、原始标签 |
| `structures.csv` | 3,100 | `383698423aeb8494bc3aa97d8363d0d8a90539206d7b42d7d00f1217bd457d2e` | 唯一结构级标签 |
| `label_evidence.csv` | 24,668 条 | `7c92066c9c8ab830cefe97322be2e05f34972dae19e7b7f76a3e0de69db2067a` | 旧标签 + 新 FUNGICIDE 证据 |
| `conflicts.csv` | 42 结构 / 110 记录 | `384a5d5d6e8d28b5b863aa88b3bad34332fe9cad4f7653769310c23173edf8c0` | 正负冲突，不进入普通监督训练 |
| `unknown_records.csv` | 2,618 条 | `8aa6a17b108c1b0138f408f9097437990da9aec8a7e7856e053cdeff3f8eac1c` | UNKNOWN，不当负样本 |
| `positive_without_structure.csv` | 375 条 | `26085b78728d3f52ef5e17025d56ecaf2f140217f9104a61c615d4544e758088` | 有正证据但无结构 |
| `modeling_structures_seed42.csv` | 2,078 | `c40ec9a1d2d523d3a54bfe3561a4f14a9f94dcf3f8f5acf98f53e75a1edc1f01` | 可监督结构 |
| `splits/split_seed42.json` | 1,454 / 312 / 312 | `7d5935d0357a7f839af594b8936cdeb9976df42d7420d77c0d4c013347ad6bad` | scaffold split |

合并后核心统计：

- 唯一结构：3,100；
- 可监督：2,078（726 正 / 1,352 负）；
- UNKNOWN：980 结构 / 2,618 记录；
- CONFLICT：42 结构 / 110 记录；
- 多成分可监督结构：260（正 168）；
- 无结构正记录：375。

这些文件属于受限原始数据的派生表，同样**不提交 GitHub**。

## 3. 哪些内容会提交

会提交：

- `src/`、`scripts/`、`tests/`：代码和测试；
- `docs/`：架构、数据来源、许可和复现说明；
- `experiments/merged_fungicide/scripts/`：构建、训练、评估、demo 代码；
- `experiments/merged_fungicide/configs/`：实验配置；
- `experiments/merged_fungicide/reports/`：数据审计和报告；
- `experiments/merged_fungicide/FINAL_REPORT.md`；
- `experiments/merged_fungicide/metrics/`：聚合指标（不含逐分子 SMILES）；
- `experiments/merged_fungicide/figures/`：聚合图；
- `experiments/merged_fungicide/appf_like/reports/`、`metrics/`、`figures/`；
- `experiments/merged_fungicide/appf_like/feature_spec.json`、`manifest.json`；
- `README.md`、`pyproject.toml`、`.gitignore`。

不会提交：

- `data/` 下的原始/派生 CSV、NPY、embedding；
- `shujv/pesticide-data/` 的原始/处理数据；
- `experiments/**/data/`、`splits/`、`features/`、`checkpoints/`、
  `models/`、`runs/`、`cache/`、`predictions/`、`logs/`；
- `models/`、`*.cbm`、`*.pt`、`*.npy`、`*.npz`；
- 用户 Excel；
- 论文 PDF/DOCX；
- `.venv`、缓存和 IDE 文件。

## 4. 本地复现步骤

1. 从 PPDB 获取并生成 `data/training/ppdb_labeled.csv`；
2. 从 `shujv/pesticide-data` 获取 D2.1 的四个文件；
3. 校验 SHA256（见上表）；
4. 运行：

```bash
cd /d/works/yuce
./.venv/Scripts/python.exe \
  experiments/merged_fungicide/scripts/build_dataset.py
./.venv/Scripts/python.exe \
  experiments/merged_fungicide/scripts/make_splits.py
./.venv/Scripts/python.exe \
  experiments/merged_fungicide/scripts/prepare_features.py
./.venv/Scripts/python.exe \
  experiments/merged_fungicide/scripts/train_catboost.py \
  --split-seeds 42,43,44 --feature-sets P24,EXT --bags 25 \
  --save-models-split 42
./.venv/Scripts/python.exe \
  experiments/merged_fungicide/scripts/analyze_appf.py
```

生成文件会落在 `experiments/merged_fungicide/` 下；这些文件被 `.gitignore`
排除。

## 5. 如果确实需要把数据放到 GitHub

只有在以下条件全部满足时才建议这样做：

1. 确认 PPDB、EPA、Canada、PubChem 等来源允许再分发；
2. 对 `LICENSE_UNKNOWN` 的来源取得书面许可；
3. 使用 **private** 仓库，不要 public；
4. 使用 Git LFS，不要直接提交 CSV/模型；
5. 在仓库 README 和 `DATA_PROVENANCE.md` 中保留来源、许可、SHA256 和
   修改说明。

在许可未确认前，默认做法是：**代码和文档公开，原始数据留在本地/机构存储。**

## 6. 数据申请模板

需要真实实验数据时见：

- `experiments/merged_fungicide/appf_like/reports/DATA_REQUEST_TEMPLATE.md`
- `experiments/merged_fungicide/appf_like/reports/assay_template.csv`

最低建议：300–500 个化合物、3 个病原、100–200 个明确无活性/弱活性化合物、
100 个盲测外部化合物，同实验条件、同终点、保留失败数据。
