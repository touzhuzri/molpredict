# 训练数据来源与标签定义

## 实际使用的数据

文件：`ppdb_labeled.csv`

| 项目 | 数量 |
| --- | --- |
| 总记录 | 1827 |
| 阳性（杀菌剂） | 433 |
| 阴性（除草剂/杀虫剂） | 1394 |
| 因混合类别排除 | 127 |
| 无 SMILES | 143 |
| SMILES 解析失败 | 15 |
| 按 InChIKey 去重排除 | 12 |

来源：**Pesticide Properties Database (PPDB)**，University of Hertfordshire
<https://sitem.herts.ac.uk/aeru/ppdb/en/>

抓取时间：2026-11-03。使用 `scripts/fetch_ppdb.py` 抓取三个公开分类列表及其详情页：

- `atoz_fung.htm`：638 条记录
- `atoz_herb.htm`：860 条记录
- `atoz_insect.htm`：784 条记录

每条记录保留：名称、SMILES、PPDB 编号、类别、InChIKey。

## 标签定义

- **阳性 label=1**：只出现在 PPDB fungicide 分类列表中的化合物。
- **阴性 label=0**：只出现在 PPDB herbicide 或 insecticide 分类列表中的化合物。
- **排除**：同时出现在 fungicide 和其他类别中的化合物（mixed），以及没有可用结构的记录。
- 没有使用无标签化合物凑阴性样本。

这是**农药分类标签**（是否登记/归类为杀菌剂），不是实验测定的杀菌活性值。

## 与 APPf 论文数据的关系

APPf 论文（DOI: 10.1016/j.aac.2025.10.005）的 489 阳 / 6409 阴数据集未公开：

- 补充材料 `mmc1.docx` 仅包含模型性能表、24 个描述符定义和模型参数；
- 论文中的 OneDrive / 服务器未提供可下载的原始分子列表；
- 因此使用同一论文数据源之一的 PPDB 重建可比标签，规模小于原文。

另调查过 ChEBI 的 fungicide/herbicide/insecticide 角色，但 3-star 结构中仅 35 个 fungicide，未采用。

## 使用限制

- 模型输出是统计预测分数，不是真实实验活性概率。
- 阴性样本是“已知非杀菌剂的农药”，不是“对所有真菌都无活性”的实验结论。
- 数据存在类别不平衡（约 1:3.2），训练使用 `auto_class_weights="Balanced"`。
