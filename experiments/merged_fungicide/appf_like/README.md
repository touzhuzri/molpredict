# appf_like

APPf 风格的 CatBoost 杀菌剂-like 原理样机。所有输出都在本目录，旧数据、
新数据和其他实验只读。

## 快速开始

```bash
cd /d/works/yuce
./.venv/Scripts/python.exe -m streamlit run \
  experiments/merged_fungicide/scripts/demo_app.py
```

打开 `http://localhost:8501`。首次评分会计算 PaDEL 24 描述符，通常几秒。

## 文档

| 文件 | 内容 |
| --- | --- |
| `reports/APPF_LIKE_REPORT.md` | 完整报告：数据、模型、结果、限制、申请资源 |
| `reports/DEMO_ONE_PAGER.md` | 给老师的一页说明 |
| `reports/results_tables.md` | 自动生成的结果表 |
| `reports/DATA_REQUEST_TEMPLATE.md` | 向老师申请真实活性数据的字段模板 |
| `reports/assay_template.csv` | 实验记录 CSV 模板 |
| `metrics/summary.json` | 全部汇总指标、AD、置换、source shift |

## 结果摘要

| 模型 | 测试 AUC | 测试 PR-AUC | 测试 F1 | 测试 MCC |
| --- | ---: | ---: | ---: | ---: |
| P24（24 PaDEL，APPf 复刻） | 0.737 ± 0.062 | 0.604 ± 0.039 | 0.639 ± 0.049 | 0.374 ± 0.132 |
| EXT（PaDEL+指纹，demo） | **0.872 ± 0.018** | **0.770 ± 0.032** | **0.744 ± 0.012** | **0.581 ± 0.041** |
| EXT_OLD_ONLY（来源消融） | 0.828 ± 0.045 | 0.721 ± 0.053 | 0.663 ± 0.026 | 0.463 ± 0.063 |

- 3 个 scaffold split；每个 split 25 个平衡 CatBoost 模型。
- split 42 EXT：AUC 0.853（95% CI 0.809–0.892），Top-10% precision 0.903。
- 旧数据训练 → 仅新数据正：AUC 0.821 ± 0.025。
- 50 次标签置换：p = 0.020。
- 适用域 Tanimoto ≥0.30：覆盖 74.4%，域内 AUC 0.884。

**边界**：这是“杀菌剂用途/登记证据”的 like 模型，不是杀菌活性模型。

## 复现

```bash
cd /d/works/yuce

./.venv/Scripts/python.exe \
  experiments/merged_fungicide/scripts/train_catboost.py \
  --split-seeds 42,43,44 --feature-sets P24,EXT --bags 25 \
  --save-models-split 42

./.venv/Scripts/python.exe \
  experiments/merged_fungicide/scripts/train_catboost.py \
  --split-seeds 42,43,44 --feature-sets EXT --bags 25 --old-only \
  --save-models-split 42,43,44

./.venv/Scripts/python.exe \
  experiments/merged_fungicide/scripts/analyze_appf.py
```

## 目录

| 路径 | 内容 |
| --- | --- |
| `models/EXT/split42/` | 25 个 demo 模型 |
| `models/P24/split42/` | 25 个 APPf 复刻模型 |
| `models/EXT_OLD_ONLY/` | 来源消融模型 |
| `predictions/` | val/test/外部集逐结构预测 |
| `metrics/` | 逐 split 指标、汇总、AD、置换 |
| `figures/` | 8 张报告图 |
| `features/` | 训练集 Morgan 指纹、AD 阈值 |

## 脚本

| 脚本 | 作用 |
| --- | --- |
| `scripts/appf_features.py` | 24 PaDEL + RDKit + Morgan/MACCS 特征 |
| `scripts/appf_eval.py` | 校准、ECE、bootstrap、指标 |
| `scripts/train_catboost.py` | 25 个平衡子集 CatBoost |
| `scripts/analyze_appf.py` | 汇总、AD、source shift、置换、图 |
| `scripts/demo_app.py` | Streamlit demo |
| `scripts/predict_excel.py` | 批量给 Excel 化合物清单打分 |
