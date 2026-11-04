# APPf 风格杀菌剂-like 原理样机报告

生成目录：`D:\works\yuce\experiments\merged_fungicide\appf_like`  
日期：2026-11-04  
数据来源：`merged_fungicide` 的旧 PPDB + 新 D2.1 合并数据  
参考论文：Jin et al., *Advanced Agrochem* 4 (2025) 373–381,
DOI 10.1016/j.aac.2025.10.005（APPf）

## 0. 一句话结论

现有资源**足够做出一个与 APPf 形式接近、并且有可重复预测信号**的原理样机：

- 严格复刻 24 个 PaDEL 描述符 + CatBoost + 25 个平衡子集：测试 ROC-AUC
  **0.737 ± 0.062**；
- 在同一流程上加入 12 个 RDKit 描述符和 Morgan/MACCS 指纹：测试 ROC-AUC
  **0.872 ± 0.018**，split 42 单次 **0.853**（95% bootstrap CI
  0.809–0.892）；
- 用“只训旧数据、预测仅新数据正结构”的 source-shift 检查：
  ROC-AUC **0.821 ± 0.025**；
- 50 次训练标签置换检验：经验 **p = 0.020**；
- 但同时必须明确：这仍然是“杀菌剂-like / 用途证据排序”，**不是实测
  杀菌活性，也不是病原特异性模型**。

这个样机足以向老师证明“流程、信号和工程化 demo 都已经跑通”，并据此申请
真实实验数据和人工标注资源；不能据此宣称已经解决杀菌活性预测。

## 1. 预测目标与边界

### 1.1 标签是什么

- 旧 PPDB：`label=1` 表示出现在 PPDB fungicide 列表；`label=0` 表示只出现
  在 herbicide/insecticide 列表。**阴性不是“对所有真菌无活性”的实验结论。**
- 新 D2.1：`FUNGICIDE=POSITIVE` 表示有杀菌剂用途/登记证据；新数据没有
  明确阴性。
- UNKNOWN 不进入二分类损失，也没有转成 0；CONFLICT 不进入普通监督训练。
- 多成分产品证据不分配给单个成分。

### 1.2 因此模型输出是什么

输出是一个 **0–1 的 fungicide-likeness / priority score**，用于排序和富集，
不是以下任何一项：

- 实验测得的 MIC / EC50 / 抑制率；
- 对某个病原的活性概率；
- 田间防治效果；
- 经过外部盲测校准的真实概率。

## 2. 数据与划分

| 项目 | 数量 |
| --- | ---: |
| 合并记录 | 6,592 |
| 唯一可解析结构 | 3,100 |
| 可监督结构 | 2,078 |
| 正 / 负 | 726 / 1,352 |
| 结构级 UNKNOWN | 980 |
| 结构级 CONFLICT | 42（110 条记录） |
| 多成分可监督结构 | 260（正 168） |

- 结构分组：同一规范结构只在一个 split；Murcko scaffold 相同的一组结构
  整体分配；6 个 ≥20 个结构的大 scaffold 强制放入训练集。
- 三个 scaffold split seed：42/43/44；每个测试集 312 个结构，正样本分别为
  128 / 116 / 113。
- 本子实验的所有模型都只使用 2,078 个可监督结构；UNKNOWN 和 CONFLICT
  没有进入二分类训练。

## 3. 模型

### 3.1 P24：严格 APPf 风格复刻

- 输入：原项目/APPf 使用的 **24 个 PaDEL 描述符**；
- 模型：CatBoost，`Logloss`，`depth=6`，`lr=0.03`，`l2=3`；
- 每个 split 训练 **25 个平衡子集模型**：每个子集使用全部训练正样本和等量
  随机负样本，概率取平均；
- 早停只看验证 ROC-AUC；阈值由验证集最大 F1 选择；测试集只评估一次。

### 3.2 EXT：用于 demo 的扩展模型

- 输入：24 PaDEL + 12 个 RDKit 描述符 + Morgan(2048) + MACCS(167)，共
  2,251 维；
- 其余流程、25 个平衡子集、早停、阈值、测试协议与 P24 完全相同；
- 这是 APPf 流程的**扩展**，不是论文原模型的严格复刻。公开报告中将 P24 和
  EXT 分开列，避免把扩展性能说成论文复现。

### 3.3 EXT_OLD_ONLY：来源消融 / source shift

- 只用旧 PPDB 有标签的结构训练，去掉所有“仅新数据正”的训练结构；
- 验证集也只用旧标签子集做早停；
- 用来检查模型能否把“仅新数据正”的结构排到前面，作为外部化学空间平移的
  近似检查。

## 4. 评估协议

- scaffold split，train/val/test = 70/15/15；
- 3 个 split seed（42/43/44）；
- 每个 split 25 个平衡 CatBoost 模型；
- 早停和阈值只用验证集；
- test 不参与特征选择、超参数选择、epoch 选择或模型回选；
- P24 vs EXT 的选择依据是**验证集 AUC**（0.794 vs 0.914），不是测试集；
- 阈值在验证集校准后的概率上按最大 F1 选择；
- 校准：在验证集上拟合 isotonic regression，再应用到测试集；
- 不确定性：split 间标准差；split 42 测试集 2,000 次 bootstrap 95% CI；
- 显著性：训练标签置换 50 次。

## 5. 结果

### 5.1 主结果（3 个 scaffold split）

| 模型 | 测试 ROC-AUC | 测试 PR-AUC | 测试 F1 | 测试 MCC | 验证 AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| P24（24 PaDEL，APPf 复刻） | 0.737 ± 0.062 | 0.604 ± 0.039 | 0.639 ± 0.049 | 0.374 ± 0.132 | 0.794 |
| EXT（PaDEL+FP，demo） | **0.872 ± 0.018** | **0.770 ± 0.032** | **0.744 ± 0.012** | **0.581 ± 0.041** | **0.914** |
| EXT_OLD_ONLY（旧数据训练） | 0.828 ± 0.045 | 0.721 ± 0.053 | 0.663 ± 0.026 | 0.463 ± 0.063 | 0.884 |

EXT 相对 P24 的提升很大，说明 24 个 PaDEL 描述符本身信息有限；Morgan/MACCS
补足了子结构和官能团信息。这个提升不是在测试集上挑出来的：验证集上 EXT
也明显高于 P24。

### 5.2 split 42 单次结果（demo 模型）

| 指标 | 数值 |
| --- | ---: |
| 测试结构 | 312（128 正 / 184 负） |
| 阈值（验证集最大 F1，校准后） | 0.4615 |
| ROC-AUC | **0.8526**（95% CI 0.809–0.892） |
| PR-AUC | **0.7492**（95% CI 0.668–0.822） |
| F1 | **0.7300**（95% CI 0.664–0.788） |
| Precision | 0.711 |
| Recall | 0.750 |
| MCC | 0.534 |
| Brier | 0.158 |
| ECE（10 bins） | 0.089 |
| 混淆矩阵 | TN 145 / FP 39 / FN 32 / TP 96 |

### 5.3 筛选富集

按校准分数从高到低排序：

| 策略 | 选中数 | 命中正样本 | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| Top 10% | 31 | 28 | **0.903** | 0.219 |
| Top 20% | 62 | 50 | **0.806** | 0.391 |

这比“随机抽”更接近实际筛选场景：如果只送 Top 10% 去做实验，测试集中
约 90% 是正样本。但测试正样本只有 128 个，Top-k 的置信区间仍然很宽。

### 5.4 来源平移（旧数据训练 → 仅新数据正）

| split | ROC-AUC | PR-AUC | F1 | MCC | Top-10% precision |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.802 | 0.730 | 0.700 | 0.464 | 0.851 |
| 43 | 0.849 | 0.803 | 0.732 | 0.538 | 0.941 |
| 44 | 0.811 | 0.755 | 0.639 | 0.455 | 0.913 |
| 均值 | **0.821 ± 0.025** | 0.763 | 0.690 | 0.486 | 0.902 |

正样本是全部 293 个“只有新数据正证据”的结构；负样本是未参与旧数据训练的
旧 PPDB 负结构。这不是独立实验外部集，但比普通 test 更接近“新化学空间”。

### 5.5 标签置换检验

- 固定 600 次迭代、同一 split 42，观察值：**0.851**；
- 50 次训练标签打乱后的 null：**0.502 ± 0.049**；
- 经验 **p = 0.0196**（1/51 下限）；
- 说明测试集上的排序信号不是由随机标签或测试集泄漏造成的。

### 5.6 适用域（applicability domain）

以 Morgan 最大 Tanimoto 相似度到训练集衡量：

- 宽松阈值 0.193（验证集 5% 分位）：覆盖 95.8% 测试结构；
  in-domain AUC 0.852，out-of-domain AUC 0.825；
- 严格标记 0.30（常用经验阈值）：覆盖 74.4%（232/312）；
  in-domain AUC **0.884**，out-of-domain AUC **0.735**。

因此 demo 中把 Tanimoto <0.30 标为“超出适用域，降低置信度”。这不是说
out-of-domain 一定错，而是提示外推风险。

### 5.7 过拟合检查

split 42 EXT 的 25 模型集成：

- train AUC：0.993；
- validation AUC：0.911；
- test AUC：0.853；
- train − test：0.140。

CatBoost 对训练集拟合很强，但 scaffold 测试仍有 0.85 左右，说明不是单纯
记忆；不过 0.14 的差距也说明正式应用前需要更多数据和正则化/外部验证。

## 6. 与 APPf 和旧深度模型的对比

| 模型 | 测试 AUC | 说明 |
| --- | ---: | --- |
| P24（24 PaDEL + CatBoost） | 0.737 | 最接近 APPf 的形式；论文报告的外部 AUC 0.7523，量级接近，但任务/数据/划分不同，不能直接等同 |
| EXT（PaDEL + 指纹 + CatBoost） | 0.872 | 本报告的 demo 模型；不是论文原模型 |
| Deep C（SMILES + PaDEL） | 0.789 | 旧 merged_fungicide 的深度模型，split 42、3 seeds |
| Deep A（SMILES） | 0.759 | 同上 |
| Deep B（SMILES + PaDEL） | 0.782 | 同上 |

结论：

1. 严格 APPf 风格模型在本地数据上只有中等信号，和论文报告的外部 AUC 同一
   量级；
2. 加入指纹后信号显著增强，说明“24 个 PaDEL 描述符”不是当前数据的瓶颈
   上限；
3. 这不证明指纹模型能预测活性，只证明结构—用途标签之间存在可学习的
   scaffold 泛化信号。

## 7. Demo 使用方法

在仓库根目录运行：

```bash
cd /d/works/yuce
./.venv/Scripts/python.exe -m streamlit run \
  experiments/merged_fungicide/scripts/demo_app.py
```

浏览器打开 `http://localhost:8501`。

Demo 功能：

- 粘贴 SMILES（支持 `name<TAB>SMILES`）；
- 上传 CSV（含 `smiles` 列）或 SDF；
- 批量输出 priority score、阈值判断、适用域标记、25 个模型的分歧度；
- 对单个分子显示 SHAP 贡献；
- 提供“随机 12 个 held-out 测试分子”按钮，避免只展示挑出来的成功案例；
- 内置 model card、ROC/PR、校准、来源平移、置换检验和化学空间图。

示例（这些分子多数在训练数据中，只用于演示界面，不作为验证）：

| 分子 | 校准分数 | 说明 |
| --- | ---: | --- |
| Tebuconazole | 1.000 | 已知杀菌剂，训练数据内 |
| Chlorothalonil | 0.814 | 已知杀菌剂，训练数据内 |
| Azoxystrobin | 0.737 | 已知杀菌剂，训练数据内 |
| Carbendazim | 0.571 | 已知杀菌剂，训练数据内 |
| Atrazine | 0.018 | 已知除草剂，旧负样本 |

注意：Glyphosate 在当前合并数据中是 UNKNOWN，模型会给较高分；这不是标签
错误，而是说明“没有真实阴性/活性数据”时模型会外推。这个例子应作为申请
人工标注和实验数据的理由，而不是 demo 的成功案例。

## 8. 这个样机证明了什么

### 8.1 已证明

- 数据合并、标签隔离、scaffold 划分、特征、CatBoost、校准、适用域、demo
  的完整流程可运行；
- 在未见过的 scaffold 测试集上有稳定高于随机的排序信号（AUC 0.87，
  置换 p=0.02）；
- 在“旧数据训练 → 新数据正结构”的来源平移检查中仍有 AUC 0.82；
- 24 PaDEL 的 APPf 风格基线可复现到与论文外部 AUC 同一量级；
- 加入指纹能显著提升内部排序，说明 pipeline 不是只能跑 24 个弱描述符。

### 8.2 没有证明

- 没有证明对任何病原的杀菌活性；
- 没有证明田间防效；
- 没有真实外部实验盲测；
- 没有病原特异性标签；
- 没有把 UNKNOWN 变成可靠阴性；
- 没有证明高分分子一定能在实验中命中。

### 8.3 主要限制

- 负样本是“只出现在 PPDB 除草/杀虫列表”的类别归属，不是实验非活性；
- 新数据没有明确阴性，正样本主要是登记/用途证据；
- 42 个结构有正负冲突，已排除；980 个结构 UNKNOWN，未用于二分类；
- 168 个可监督正结构是多成分产品；标签属于产品/配方证据，不能分配给
  单个有效成分；
- 可监督结构只有 2,078 个，split 42 测试正样本只有 128 个，Top-k 和子集
  指标仍有明显抽样波动；
- scaffold split 阻止同一 scaffold 跨 split，但不能排除相似类似物/同源
  产品带来的化学空间邻近；
- demo 中的已知分子多数在训练数据中；它们只用于演示界面，不作为验证；
- Glyphosate 在当前数据中是 UNKNOWN，模型给高分，说明没有真实阴性时模型
  会外推；这类案例正是需要人工标注和实验数据的地方。

## 9. 向老师申请的数据与人力

可直接交给老师的字段模板：`reports/DATA_REQUEST_TEMPLATE.md` 和
`reports/assay_template.csv`。

### 9.1 数据

建议先要“回顾性数据”，成本最低：

- 实验室已有的杀菌剂筛选记录：化合物 ID/SMILES、病原、浓度、抑制率/
  MIC/EC50、实验日期、重复；
- 失败/无活性化合物也要，不能只给成功化合物；
- 至少 300–500 个化合物，3 个病原：
  - *Botrytis cinerea*
  - *Fusarium graminearum*
  - *Magnaporthe oryzae*
- 同一实验条件、同一浓度梯度、同一终点；
- 100–200 个明确无活性/弱活性化合物；
- 100 个盲测外部化合物，不参与训练；
- 如果做田间/温室，另设外部集。

### 9.2 人力

- 0.5–1 人 × 3 个月：数据整理、结构标准化、标签复核、负样本筛选；
- 0.5 人 × 3–6 个月：微生物实验；
- 0.1–0.2 人：导师/植保专家确定病原、实验标准和标签规则；
- 计算资源：现有 8 GB GPU 足够 demo；正式训练可用 CPU/单卡。

### 9.3 建议的验证闭环

1. 用当前 demo 对化合物库排序；
2. 选 Top 10–20% + 随机对照，送同一病原面板实验；
3. 把实验结果作为新标签；
4. 重新训练，报告 prospective hit rate，而不是只报告 retro AUC；
5. 如果 Top-k 命中率显著高于随机对照，再扩大规模。

## 10. 复现

```bash
cd /d/works/yuce

# 1. 正式 CatBoost（P24 + EXT，3 splits，25 bags）
./.venv/Scripts/python.exe \
  experiments/merged_fungicide/scripts/train_catboost.py \
  --split-seeds 42,43,44 --feature-sets P24,EXT --bags 25 \
  --save-models-split 42

# 2. 来源消融（旧数据训练）
./.venv/Scripts/python.exe \
  experiments/merged_fungicide/scripts/train_catboost.py \
  --split-seeds 42,43,44 --feature-sets EXT --bags 25 --old-only \
  --save-models-split 42,43,44

# 3. 汇总、校准、适用域、来源平移、置换检验、图
./.venv/Scripts/python.exe \
  experiments/merged_fungicide/scripts/analyze_appf.py

# 4. Demo
./.venv/Scripts/python.exe -m streamlit run \
  experiments/merged_fungicide/scripts/demo_app.py
```

## 11. 产物

| 路径 | 内容 |
| --- | --- |
| `appf_like/metrics/runs.csv` | P24/EXT 逐 split 指标 |
| `appf_like/metrics/runs_OLD_ONLY.csv` | 来源消融指标 |
| `appf_like/metrics/summary.json` | 汇总、AD、置换、source shift |
| `appf_like/metrics/EXT_split42_selected.json` | demo 模型完整指标 |
| `appf_like/metrics/isolation_audit.json` | split 结构/scaffold 重叠审计 |
| `appf_like/predictions/` | val/test/外部集逐结构预测 |
| `appf_like/models/EXT/split42/` | 25 个 demo 模型 |
| `appf_like/models/P24/split42/` | 25 个 APPf 复刻模型 |
| `appf_like/models/EXT_OLD_ONLY/` | 来源消融模型 |
| `appf_like/figures/` | 8 张报告图 |
| `appf_like/reports/results_tables.md` | 机器生成结果表 |

## 12. 最终建议

**先做 demo，再要数据，不要先承诺活性模型。**

对老师可以这样表述：

> “我们已经把 APPf 风格的 24-PaDEL CatBoost 流程复刻出来，并做了一个可
> 交互的 demo。严格 24 描述符版本在 scaffold 测试上 AUC 约 0.74，与论文
> 外部 AUC 同一量级；加入指纹后内部 AUC 约 0.87，来源平移约 0.82，置换
> 检验 p=0.02。它证明流程和信号存在，但标签是用途/登记证据，不是活性。
> 现在缺的是同一病原、同一实验条件下的正负样本和外部盲测。给我 300–500
> 个实测化合物、3 个病原和 100 个盲测，我就能把它从原理样机推进到可验证
> 的活性模型。”
