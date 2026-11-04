# merged_fungicide

新旧数据合并后的独立杀菌剂用途预测实验。所有新数据、模型权重、日志和报告都
在本目录内；旧项目和 `shujv/pesticide-data` 只读。

## 目录

| 路径 | 内容 |
| --- | --- |
| `reports/data_audit.md` | 合并、标签、冲突、划分和限制 |
| `FINAL_REPORT.md` | 最终报告（结果、失败、结论） |
| `data/merged_fungicide_records.csv` | 6,592 条合并记录，保留原始 ID/来源/标签 |
| `data/structures.csv` | 3,100 个唯一结构及结构级标签 |
| `data/label_evidence.csv` | 旧标签来源 + 新 FUNGICIDE 证据 |
| `data/unknown_records.csv` | UNKNOWN 记录（不当负样本） |
| `data/conflicts.csv` | 42 个冲突结构、110 条记录 |
| `data/positive_without_structure.csv` | 375 条无结构正记录 |
| `data/modeling_structures_seed*.csv` | 2,078 个可监督结构 |
| `splits/split_seed*.json` | 结构分组 scaffold split |
| `features/` | ChemBERTa token states、PaDEL 24、特征清单 |
| `configs/A.yaml`、`B.yaml`、`C.yaml` | SMILES、SMILES+PaDEL、辅助学习 |
| `scripts/` | 构建、划分、特征、训练、分析代码 |
| `checkpoints/` | 每个 run 的权重、config、history、预测 |
| `metrics/` | 逐 run 指标、汇总表、置换检验 |
| `figures/` | 结果图 |

## 数据规则摘要

- 旧 PPDB：1,827 条；阳性 433，阴性 1,394。阴性定义是“只出现在
  herbicide/insecticide 列表”，不是实验非活性。
- 新 D2.1：4,765 个候选；结构可解析 2,924；FUNGICIDE POSITIVE 1,008，
  UNKNOWN 3,757，明确阴性 0。
- 合并：6,592 条记录，4,751 条可解析，3,100 个唯一结构。
- 可监督：2,078 个结构（726 正 / 1,352 负）。
- 42 个结构同时有正、负证据，标为 CONFLICT，不进入普通监督损失。
- UNKNOWN 不进入二分类损失，也没有转成 0。
- 负样本全部来自旧 PPDB；新数据没有负标签。

## 模型

- A：冻结 ChemBERTa token states → 4 层统一 Transformer → CLS 分类。
- B：A + 24 个 PaDEL descriptor token。
- C：B + “SMILES token states → 24 PaDEL”的 masked MSE 辅助任务；
  辅助集只含训练结构以及与验证/测试 scaffold 不重叠的 UNKNOWN/CONFLICT
  结构。
- 所有模型随机初始化，不加载旧项目分类器权重。
- 早停只看验证 ROC-AUC；阈值由验证集最大 F1 选择；测试集只评估一次。

## 关键结果（split 42，3 seeds）

| 实验 | 测试 AUC | 测试 PR-AUC |
| --- | --- | --- |
| A SMILES | 0.7593 ± 0.0173 | 0.6840 ± 0.0163 |
| B SMILES+PaDEL | 0.7818 ± 0.0397 | 0.7028 ± 0.0557 |
| C B+辅助 | **0.7886 ± 0.0161** | **0.7106 ± 0.0275** |
| B_OLD_ONLY | 0.7623 ± 0.0146 | 0.6533 ± 0.0262 |

C 的均值最高但相对 B 仅 +0.0068；测试期 PaDEL 置换只下降 <0.002，
说明 24 个描述符基本没有被主分类路径使用。新数据对“仅新数据正”子集有
方向性帮助（B 0.827 vs B_OLD 0.774），但没有改善旧正子集，且总体增益
不稳健。详见 `FINAL_REPORT.md`。

## 复现

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
./.venv/Scripts/python.exe experiments/merged_fungicide/scripts/padel_permutation.py
```

实际结果与限制见 `FINAL_REPORT.md`。
