# molpredict V3 实验报告

> 本报告记录 V3 统一多模态 Transformer 的实际训练与评估结果。所有指标都是
> PPDB 农药分类标签上的模型指标，不是实验抗真菌活性。

## 0. 新增与修改文件

| 文件 | 说明 |
| --- | --- |
| `src/molpredict/v3_encoders.py` | ChemBERTa/ConfSeq 加载、token 编码、缓存与哈希 |
| `src/molpredict/v3_model.py` | 统一 token 序列、descriptor token、共享 Transformer |
| `src/molpredict/v3_data.py` | V3 缓存校验、dataset/collate、训练集标准化、泄漏审计 |
| `src/molpredict/v3_train.py` | 冻结/部分解冻训练、早停、阈值、checkpoint 与指标保存 |
| `src/molpredict/v3_predict.py` | SMILES + ConfSeq + 24 PaDEL 的预测接口 |
| `scripts/run_v3_experiments.py` | full / 三组消融 / full_ft 实验入口 |
| `scripts/aggregate_v3_runs.py` | 合并并行输出、汇总 summary/stability/audit |
| `tests/test_v3.py` | 形状、mask、消融、标准化、冻结、保存/加载等测试 |
| `src/molpredict/cli.py` | 增加 `v3-predict` 子命令 |
| `docs/v3_architecture.md` | V3 架构、输入、冻结策略、checkpoint 与许可证 |
| `docs/v3_report.md` | 本报告 |
| `docs/DATA_SOURCES.md` | 补充 V3 数据/缓存/划分复用与泄漏状态 |
| `README.md` | 增加 V3 架构、命令、结果入口 |
| `experiments/v3/` | 缓存、划分、run 产物、汇总与审计 |

V2 代码、V2 结果和 `models/` 下的 CatBoost 基线均未修改或覆盖。

## 1. 数据与划分

- 训练文件：`data/training/confseq.csv`，1823 个分子（431 阳性 / 1392 阴性）。
- 阳性 = PPDB fungicide 列表；阴性 = PPDB herbicide 或 insecticide 列表。
- 标签来自 PPDB 分类，不是 EC50/MIC/实验抑制率。
- 复用 V2 的 Murcko scaffold 三套划分（seed 42/43/44）：

| seed | train | val | test | test 阳性 |
| --- | --- | --- | --- | --- |
| 42 | 1332 | 168 | 323 | 80 |
| 43 | 1320 | 167 | 336 | 82 |
| 44 | 1295 | 208 | 320 | 83 |

- 标准化/缺失值中位数只由训练集拟合；早停和阈值只在验证集；测试集只评估一次。
- `experiments/v3/leakage_audit.json` 检查结果：InChIKey 重复 0、canonical SMILES 重复 0，
  train/val/test 之间索引、InChIKey、Murcko scaffold 重叠均为 0（seed 42/43/44）。
- 预训练语料重叠：**unknown**。没有 ZINC/ConfSeq 完整预训练分子清单，无法逐分子审计。

## 2. 模型与训练配置

V3 的详细架构见 [`v3_architecture.md`](v3_architecture.md)。

核心配置：

```text
d_model=128, n_layers=4, n_heads=4, ff_mult=2, dropout=0.3（正式）
max_smiles_len=320, max_confseq_len=512
SMILES: ChemBERTa token states [L,768]
ConfSeq: official token states [L,256]
PaDEL: 24 independent descriptor tokens
fusion: [CLS] + SMILES + ConfSeq + 24 descriptors → shared Transformer → CLS → 1 logit
pooling: CLS（可选 cls_mean 仅作为代码选项，未用于正式结果）
```

第一阶段冻结两个预训练编码器；第二阶段 `full_ft` 只解冻 ChemBERTa 最后 2 层。
优化器 AdamW，正式冻结实验使用融合头 lr=3e-4、dropout=0.3、batch=16、最多 20 epochs、
patience=5（初始 lr=1e-3/dropout=0.2 pilot 见第 7 节）；`full_ft` 编码器 lr=2e-5，
融合头 lr=3e-4、batch=32、最多 12 epochs、patience=4，三套划分均已运行。
训练使用 `BCEWithLogitsLoss(pos_weight=neg/pos)`，按验证集 ROC-AUC 早停。

## 3. 实验矩阵

| 名称 | 输入 |
| --- | --- |
| `full` | SMILES + ConfSeq + 24 PaDEL |
| `no_smiles` | ConfSeq + 24 PaDEL |
| `no_confseq` | SMILES + 24 PaDEL |
| `no_padel` | SMILES + ConfSeq |
| `full_ft` | 同 full，但解冻 ChemBERTa 最后 2 层（seed 42/43/44） |

对照复用 V2 已有结果（同一 scaffold 划分）：A CatBoost、F、F-ft。

## 4. 结果

### 4.1 三套 scaffold 划分的测试均值

均值 ± 样本标准差（n=3）。`full_ft` 也完成三套划分，但它使用 batch=32、12 epochs、
patience=4 的不同训练配方，与冻结实验不是严格的单变量对照。阈值由每套划分的
验证集单独选择，测试集只评估一次。

| 配置 | 输入 | n | ROC-AUC | PR-AUC | F1 | Precision | Recall | MCC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `full` | SMILES + ConfSeq + PaDEL 24 | 3 | 0.7513 ± 0.0571 | 0.5014 ± 0.0589 | 0.4689 ± 0.0652 | 0.4559 ± 0.0707 | 0.5410 ± 0.2458 | 0.2981 ± 0.0544 |
| `no_smiles` | ConfSeq + PaDEL 24 | 3 | **0.8066 ± 0.0222** | **0.6024 ± 0.0695** | **0.5682 ± 0.0428** | **0.4878 ± 0.0380** | 0.6850 ± 0.0861 | **0.4033 ± 0.0576** |
| `no_confseq` | SMILES + PaDEL 24 | 3 | 0.7155 ± 0.0749 | 0.4612 ± 0.0474 | 0.4741 ± 0.0723 | 0.3755 ± 0.0562 | 0.6486 ± 0.1300 | 0.2530 ± 0.1034 |
| `no_padel` | SMILES + ConfSeq | 3 | 0.7850 ± 0.0248 | 0.5719 ± 0.0402 | 0.5495 ± 0.0126 | 0.4714 ± 0.0919 | **0.7072 ± 0.1585** | 0.3830 ± 0.0294 |
| `full_ft` | SMILES + ConfSeq + PaDEL 24，ChemBERTa 最后 2 层解冻 | 3 | 0.8191 ± 0.0210 | 0.6223 ± 0.0296 | 0.5825 ± 0.0312 | 0.4755 ± 0.0209 | 0.7581 ± 0.0948 | 0.4240 ± 0.0399 |

粗体只比较前四行的冻结实验；`full_ft` 使用不同 batch/轮数配方，单独看。

### 4.2 逐 seed 测试结果

| 配置 | seed | 验证 AUC | 测试 AUC | 测试 PR-AUC | 测试 F1 | 测试 MCC | 最佳 epoch |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `full` | 42 | 0.8199 | 0.6854 | 0.4338 | 0.4364 | 0.2435 | 20 |
| `full` | 43 | 0.7780 | 0.7809 | 0.5285 | 0.4265 | 0.2985 | 14 |
| `full` | 44 | 0.8842 | 0.7875 | 0.5419 | 0.5440 | 0.3523 | 18 |
| `no_smiles` | 42 | 0.7864 | 0.7876 | 0.5319 | 0.5279 | 0.3435 | 7 |
| `no_smiles` | 43 | 0.7801 | 0.8013 | 0.6045 | 0.5635 | 0.4079 | 9 |
| `no_smiles` | 44 | 0.8880 | 0.8309 | 0.6709 | 0.6132 | 0.4585 | 14 |
| `no_confseq` | 42 | 0.8125 | 0.7461 | 0.4602 | 0.5196 | 0.3287 | 10 |
| `no_confseq` | 43 | 0.6522 | 0.6302 | 0.4143 | 0.3907 | 0.1352 | 3 |
| `no_confseq` | 44 | 0.7573 | 0.7702 | 0.5091 | 0.5120 | 0.2952 | 13 |
| `no_padel` | 42 | 0.8053 | 0.7737 | 0.5277 | 0.5354 | 0.3583 | 6 |
| `no_padel` | 43 | 0.8026 | 0.7678 | 0.5816 | 0.5535 | 0.4156 | 9 |
| `no_padel` | 44 | 0.8602 | 0.8134 | 0.6064 | 0.5596 | 0.3751 | 12 |
| `full_ft` | 42 | 0.8258 | 0.8192 | 0.5992 | 0.5532 | 0.3839 | 10 |
| `full_ft` | 43 | 0.7661 | 0.7980 | 0.6121 | 0.5789 | 0.4245 | 5 |
| `full_ft` | 44 | 0.8606 | 0.8400 | 0.6556 | 0.6154 | 0.4637 | 9 |

`full` seed42 是唯一一个验证 AUC 与测试 AUC 相差超过 0.1 的 run；它在 epoch 20 的
训练损失已降到 0.169，验证 AUC 仍在缓慢上升，说明早停没有阻止对训练骨架的拟合。
`full_ft` 的验证/测试差分别为 +0.0065 / −0.0319 / +0.0206，三套划分方向不一致；
seed43 的验证 AUC 最低（0.7661），测试反而高于验证，最佳 epoch 只有 5。

### 4.3 测试集混淆矩阵

阈值为该划分验证集最大 F1 对应的阈值。

| 配置/seed | TN | FP | FN | TP |
| --- | --- | --- | --- | --- |
| `full` 42 | 194 | 49 | 44 | 36 |
| `full` 43 | 229 | 25 | 53 | 29 |
| `full` 44 | 138 | 99 | 15 | 68 |
| `no_smiles` 42 | 178 | 65 | 28 | 52 |
| `no_smiles` 43 | 206 | 48 | 31 | 51 |
| `no_smiles` 44 | 173 | 64 | 18 | 65 |
| `no_confseq` 42 | 172 | 71 | 27 | 53 |
| `no_confseq` 43 | 163 | 91 | 40 | 42 |
| `no_confseq` 44 | 134 | 103 | 19 | 64 |
| `no_padel` 42 | 137 | 106 | 12 | 68 |
| `no_padel` 43 | 221 | 33 | 38 | 44 |
| `no_padel` 44 | 163 | 74 | 22 | 61 |
| `full_ft` 42 | 187 | 56 | 28 | 52 |
| `full_ft` 43 | 174 | 80 | 16 | 66 |
| `full_ft` 44 | 167 | 70 | 15 | 68 |

不同 seed 的阈值差异很大（例如 `full` 42 为 0.0215，`full` 43 为 0.9852），说明
在只有 167–208 条验证集上选阈值本身就不稳定；混淆矩阵应结合阈值一起看。

## 5. 消融

三组消融只改变输入模态，保留相同的 scaffold 划分、训练轮数、lr、dropout 和融合网络：

| 名称 | 保留的输入 | 目的 |
| --- | --- | --- |
| `full` | SMILES + ConfSeq + PaDEL | 完整统一模型 |
| `no_smiles` | ConfSeq + PaDEL | 检查 SMILES token 的贡献 |
| `no_confseq` | SMILES + PaDEL | 检查 ConfSeq token 的贡献 |
| `no_padel` | SMILES + ConfSeq | 检查 24 个 descriptor token 的贡献 |

### 5.1 均值变化

| 对比 | 测试 AUC 均值 | 相对 `full` | 逐 seed ΔAUC（42/43/44） |
| --- | --- | --- | --- |
| `full` | 0.7513 | — | — |
| `no_smiles` | 0.8066 | **+0.0553** | +0.1021 / +0.0204 / +0.0434 |
| `no_confseq` | 0.7155 | −0.0358 | +0.0607 / −0.1507 / −0.0173 |
| `no_padel` | 0.7850 | +0.0337 | +0.0883 / −0.0131 / +0.0259 |

- **去掉 SMILES 在三套划分上都变好**，这是本次消融中最一致的方向。SMILES token
  序列没有给这个融合模型带来可测的增益，反而可能因长序列和冻结的 ChemBERTa 表示
  增加了过拟合。
- 去掉 ConfSeq 的均值更低，但方向不一致：seed42 反而提高 0.0607，seed43 大幅下降
  0.1507。均值差 −0.0358 主要由 seed43 驱动，不能据此断言“ConfSeq 一定有用”。
- 去掉 24 个 descriptor 的均值略高（+0.0337），但逐 seed 方向混合；descriptor 的
  作用依赖其他模态是否在场，不能简单说“有用”或“无用”。

### 5.2 部分解冻对照

`full_ft` 三套划分的测试 AUC 为 **0.8191 ± 0.0210**（PR-AUC 0.6223 ± 0.0296、
F1 0.5825 ± 0.0312、MCC 0.4240 ± 0.0399）。逐 seed 与冻结 `full` 相比分别提高
+0.1338 / +0.0171 / +0.0525，均值 +0.0678；但 `full_ft` 使用 batch=32、12 epochs、
patience=4，而冻结实验是 batch=16、20 epochs、patience=5，所以这不是严格的
“只改解冻”单变量对照。与 V2 F-ft（0.8159 ± 0.0197）基本持平（+0.0032，远小于
两者标准差），仍低于 V2 A CatBoost（0.8578 ± 0.0292）。三套划分中 seed42 的
`full_ft`（0.8192）低于同 seed 的 V2 F-ft（0.8349），seed43/44 则略高。

三组输入消融和 `full_ft` 的共同信息是：V3 的统一架构可以训练并能输出合理排序，
但在当前数据规模下，“把所有 token 都放进去”不等于更好；冻结 `no_smiles` 反而更稳，
而部分解冻 ChemBERTa 能把 `full` 拉回到与 V2 F-ft 相当的水平。

## 6. 与 V2 的比较

V2 结果直接复用同一 scaffold 划分的已保存 run（未重训）：

| 模型 | ROC-AUC | PR-AUC | F1 | MCC |
| --- | --- | --- | --- | --- |
| V2 A CatBoost（Morgan + RDKit 217） | 0.8578 ± 0.0292 | 0.6985 ± 0.0285 | 0.6111 ± 0.0513 | 0.4741 ± 0.0566 |
| V2 F attention（ChemBERTa + ConfSeq + PaDEL） | 0.7602 ± 0.0062 | 0.5120 ± 0.0887 | 0.5276 ± 0.0026 | 0.3376 ± 0.0096 |
| V2 F-ft（ChemBERTa 最后 2 层解冻） | 0.8159 ± 0.0197 | 0.6053 ± 0.0078 | 0.5805 ± 0.0455 | 0.4237 ± 0.0665 |

V3 的对照行：

| 模型 | ROC-AUC | PR-AUC | F1 | MCC |
| --- | --- | --- | --- | --- |
| V3 `full` | 0.7513 ± 0.0571 | 0.5014 ± 0.0589 | 0.4689 ± 0.0652 | 0.2981 ± 0.0544 |
| V3 `no_smiles` | 0.8066 ± 0.0222 | 0.6024 ± 0.0695 | 0.5682 ± 0.0428 | 0.4033 ± 0.0576 |
| V3 `no_confseq` | 0.7155 ± 0.0749 | 0.4612 ± 0.0474 | 0.4741 ± 0.0723 | 0.2530 ± 0.1034 |
| V3 `no_padel` | 0.7850 ± 0.0248 | 0.5719 ± 0.0402 | 0.5495 ± 0.0126 | 0.3830 ± 0.0294 |
| V3 `full_ft` | 0.8191 ± 0.0210 | 0.6223 ± 0.0296 | 0.5825 ± 0.0312 | 0.4240 ± 0.0399 |

- 冻结的 V3 `full`（0.7513）低于 V2 F（0.7602），也远低于 V2 A（0.8578）和
  V2 F-ft（0.8159）。把三种 token 放进共同注意力并没有自动带来提升。
- V3 中最好的冻结三划分配置是 `no_smiles`（0.8066），高于 V2 F（0.7602），接近但
  低于 V2 F-ft（0.8159），仍低于 CatBoost A（0.8578）。
- `full_ft` 的三划分均值（0.8191）与 V2 F-ft（0.8159）基本持平，高于 V2 F，但仍
  低于 CatBoost A（0.8578）；它使用了更长的 batch/不同的轮数配方，不能视为严格
  单变量优势。
- 因而当前 **CatBoost A 仍是最好的配置**；V3 的统一 token 级融合是可运行、可审计的
  实验架构，但没有证据表明它在 PPDB 分类任务上优于 V2 基线或 V2 的池化融合。

V3 和 V2 使用同一数据、同一 scaffold 索引、同一测试集；V2 的 A 是 CatBoost，
V2 F 是池化后融合，V2 F-ft 是 V2 架构的部分解冻。V3 的融合发生在 token 级。

## 7. 失败/未完成实验

- 初始 `full` pilot（seed42，lr=1e-3，dropout=0.2，20 epochs，patience=5）在验证集
  epoch 5 达到最佳 AUC 0.7782，测试 AUC 0.7723、PR-AUC 0.5221、F1 0.4923、
  MCC 0.2923。这个结果低于 V2 E/F，因此没有直接作为正式配置；后续正式实验比较
  更低学习率和更高 dropout。
- 正式 `full` seed42（lr=3e-4、dropout=0.3）在 epoch 20 的验证 AUC 0.8199，但测试
  AUC 仅 0.6854，验证-测试差约 0.135。这是明显的 scaffold 验证集过拟合：模型在
  训练骨架附近变好，却没有迁移到测试骨架。正式结果保留，不据此重选测试成绩。
- `no_confseq` seed43 的验证 AUC 只有 0.6522、测试 AUC 0.6302，最佳 epoch 3 后早停；
  这是所有正式 run 中最差的一次，说明去掉 ConfSeq 后该划分对初始化/优化很敏感。
- CPU 上每个 epoch 约 25–150 秒（ConfSeq 长序列使 full 较慢，`full_ft` 的 live
  ChemBERTa 约 2 分钟/epoch）；未做大规模超参数搜索，避免在测试集上反复调参。
- `full_ft`（解冻 ChemBERTa 最后 2 层）最后补跑了 seeds43/44，三套划分都有结果；
  但该对照使用 batch=32、12 epochs、patience=4，与冻结主实验的 batch=16、20 epochs、
  patience=5 不完全相同，因此只能作为探索性对照，不能作为严格单变量结论。没有为了
  “端到端”而解冻 ConfSeq 或全量微调两个编码器。

## 8. 局限性

- 标签是 PPDB 农药分类，不是实验抗真菌活性；阴性也不是“实验证明无活性”。
- 仅 431 个阳性；三套 scaffold 划分的测试集有重叠，mean±std 不是三次独立复现。
- PaDEL 24 中 11 个超大/聚合分子缺失，11 个分子部分 E-State 缺失；V3 只使用训练集
  中位数填补。
- 预训练语料重叠状态为 **unknown**：没有 ZINC/ConfSeq 预训练语料的完整分子清单。
  只确认了训练/验证/测试内部没有 InChIKey、canonical SMILES、Murcko scaffold 重叠。
- CPU 训练；融合网络本身约 0.79M 参数，但依赖两个冻结预训练编码器。`full_ft` 只解冻
  ChemBERTa 最后 2 层，未解冻 ConfSeq，也未做全量微调；使用 batch=32、12 epochs，
  与冻结实验的 batch=16、20 epochs 不完全一致。
- 超参数（4 层、128 维、lr、dropout）未做系统搜索；验证集较小（167–208 条），
  早停和阈值选择有波动。正式 full seed42 出现验证 AUC 0.8199 / 测试 AUC 0.6854
  的大差距，说明单靠该验证集选模型会过拟合；没有用测试集回选模型。
- 注意力权重只能说明模型内部的统计交互，不能解释为真实化学作用位点。
- 没有 V3 外部实验活性验证；现有外部验证仍来自旧 CatBoost 模型。
- V3 的 ConfSeq 输入必须由官方流程生成；预测接口不会自动生成 ConfSeq，也不会用
  RDKit 3D 替代。

## 9. 结论

V3 已经实现了任务要求的**单一统一多模态 Transformer**：SMILES/ConfSeq 的完整 token
序列和 24 个独立 descriptor token 进入同一条序列、同一个 4 层自注意力网络，最后只
通过 CLS 产生一个 PPDB 杀菌剂分类分数；没有三个独立模型，没有投票，也没有在融合前
池化。冻结、mask、token 顺序、标准化、阈值和测试流程都已写入 checkpoint/文档/测试。

但“统一”本身没有带来更好的指标：

- 冻结 `full` 的三划分测试 AUC 为 **0.7513 ± 0.0571**，低于 V2 F（0.7602），更低于
  V2 A CatBoost（0.8578）和 V2 F-ft（0.8159）。
- V3 内部最好的是 `no_smiles`（ConfSeq + 24 descriptors），**0.8066 ± 0.0222**；
  去掉 SMILES 在三套划分上都提高，说明当前冻结 ChemBERTa 的 SMILES token 流在这个
  小数据、长序列融合中没有带来增益。
- 去掉 ConfSeq 或不使用 descriptors 的均值分别降到 0.7155 和 0.7850，但逐 seed
  方向不一致；不能仅凭 3 套有重叠的 scaffold 划分做显著性结论。
- `full_ft`（只解冻 ChemBERTa 最后 2 层）三划分均值为 **0.8191 ± 0.0210**，高于冻结
  `full`（0.7513 ± 0.0571），与 V2 F-ft（0.8159 ± 0.0197）基本持平，但仍低于
  V2 A（0.8578 ± 0.0292）。它使用不同 batch/轮数配方，且方差仍较大
  （逐 seed 0.7980–0.8400），不能视为稳定超越。

因此，**当前最可靠的模型仍是 CatBoost A，V3 不替换基线**。V3 的贡献是提供一个可
复现、可审计的 token 级统一融合实现和一组诚实的消融结果：它证明该架构能训练、
能预测，但在 PPDB 分类任务和现有数据规模下没有超过 V2。所有测试集只评估一次，
阈值只在验证集选择，没有用测试集回选配置。

## 10. 对关键问题的明确回答

**1. 模型是否真正统一？**
是，融合和分类层面是单一网络：一条 `[CLS] + SMILES tokens + ConfSeq tokens +
descriptor tokens` 序列进入同一个 4 层 Transformer，只有一个分类头和一个输出；
没有三个独立模型、投票或先池化再拼接。需要限定的是：默认配置下两个预训练编码器
冻结，token 表示在进入融合网络前由各自编码器计算；所以它是“单一融合模型”，不是
从原始字符串到输出全参数端到端联合训练。`full_ft` 只解冻 ChemBERTa 最后 2 层，
ConfSeq 始终冻结。

**2. 是否保留 token 级表示？**
是。SMILES 使用 ChemBERTa `last_hidden_state`（最大 320 token，实测最大 298，无截断），
ConfSeq 使用官方编码器 `last_hidden_state`（最大 512，实测最大 499）；两者都没有在
进入共享 Transformer 前池化。padding 通过 key-padding mask 屏蔽，并在序列表示上清零。

**3. 24 个描述符是否作为独立 token 参与？**
是。24 个 PaDEL 特征各自变成一个 token：标准化标量经小型 MLP，加上该特征的固定
identity embedding 和 descriptor 模态 embedding；24 个位置都有效，顺序固定并保存在
`feature_order.json`。它们不是原子位置序列，也没有被假设与第 i 个 SMILES/ConfSeq
token 对应。

**4. 模态之间如何相互注意？**
在共享 Transformer 的每一层，所有非 padding token 在同一个自注意力矩阵中互相可见：
SMILES token 可以看 ConfSeq/descriptor token，反之亦然；descriptor token 之间也可
互相参考。SMILES/ConfSeq 额外有模态内可学习位置 embedding，所有 token 有模态
embedding；最终 CLS 汇总后分类。没有人为的跨模态位置对齐。

**5. 哪些冻结、哪些可训练？**
冻结主实验：ChemBERTa 和 ConfSeq 都冻结；可训练的是两个投影层、模态/位置
embedding、24 个 descriptor token 编码器、4 层融合 Transformer 和分类头。`full`
融合网络总参数约 0.79M，全部可训练。`full_ft` 额外解冻 ChemBERTa 最后 2 层
（总参数 44,893,057，可训练 14,964,865），编码器 lr=2e-5，ConfSeq 仍冻结。

**6. V3 与 V2 的性能对比？**
冻结 V3 `full` 的测试 AUC 0.7513 ± 0.0571，低于 V2 F（0.7602 ± 0.0062）、
V2 F-ft（0.8159 ± 0.0197）和 V2 A CatBoost（0.8578 ± 0.0292）。V3 最好的三划分
配置 `no_smiles` 为 0.8066 ± 0.0222，高于 V2 F，接近但低于 V2 F-ft，仍低于 A。
`full_ft` 三划分均值 0.8191 ± 0.0210，与 V2 F-ft 0.8159 ± 0.0197 基本持平，但低于
V2 A 0.8578 ± 0.0292；逐 seed 0.7980–0.8400，方差大于 V2 F-ft。
因此 V3 没有超过 V2 的 CatBoost A；A 仍是最可靠配置。

**7. 哪些输入有贡献？**
按三划分均值：去掉 SMILES 后从 0.7513 升到 0.8066（三套划分都升高），说明当前
SMILES token 流没有正贡献；去掉 ConfSeq 后降到 0.7155，但 seed42 反而升高，方向
不一致；去掉 24 descriptors 后升到 0.7850，逐 seed 方向也混合。可以确定的是
“ConfSeq + 24 descriptors、不含 SMILES”是本次最稳的 V3 输入组合；但不能把
descriptor 或 ConfSeq 的均值差当作显著因果效应。

**8. 泄漏风险？**
数据内部：`experiments/v3/leakage_audit.json` 对 seed42/43/44 均记录 InChIKey 重复 0、
canonical SMILES 重复 0，train/val/test 的索引、InChIKey、Murcko scaffold 重叠均为 0。
流程上：标准化/中位数只用训练集拟合，早停和阈值只用验证集，测试集只评估一次。
仍存在的风险：预训练语料重叠为 **unknown**（没有 ZINC/ConfSeq 完整分子清单）；三套
scaffold 测试集彼此有重叠，mean±std 不是三次独立复现；标签是 PPDB 分类而非实验活性；
V3 没有新的外部实验验证。

**9. 主要局限？**
431 个阳性、数据规模小；验证集只有 167–208 条，早停和阈值不稳定；`full` seed42
出现验证 0.8199/测试 0.6854 的大差距；`full_ft` 的 batch/轮数与冻结实验不同；
没有系统超参数搜索；CPU 训练；24 个 PaDEL 有缺失值；ConfSeq 必须由官方流程生成；
注意力权重不能解释为化学作用位点；V3 仍是实验性模型，不能替代 CatBoost 基线，也不能
作为实验活性筛选依据。

## 11. 运行命令

```bash
cd D:/works/yuce

# 安装已有依赖（V3 复用 V2 的 torch/transformers/padelpy）
pip install -e ".[confseq,v2]"

# 缓存：V3 320-token SMILES + 校验 V2 ConfSeq/PaDEL/顺序
./.venv/Scripts/python.exe scripts/run_v3_experiments.py --prepare

# 主实验 + 消融，三套 scaffold 划分
./.venv/Scripts/python.exe scripts/run_v3_experiments.py \
  --experiments full,no_smiles,no_confseq,no_padel \
  --seeds 42,43,44 --epochs 20 --patience 5 --lr 3e-4 --dropout 0.3

# 部分解冻（ChemBERTa 最后 2 层；batch=32、12 epochs、patience=4）
./.venv/Scripts/python.exe scripts/run_v3_experiments.py \
  --experiments full_ft --seeds 42,43,44 --epochs 12 --patience 4 \
  --batch-size 32 --lr 3e-4 --encoder-lr 2e-5 --dropout 0.3

# 冒烟训练（每侧 64 条、2 epochs）
./.venv/Scripts/python.exe scripts/run_v3_experiments.py \
  --experiments full --seeds 42 --quick --out experiments/v3_quick

# 并行跑不同配置时，各自写独立 --out，最后汇总（可选）
./.venv/Scripts/python.exe scripts/aggregate_v3_runs.py \
  --roots experiments/v3_formal_full experiments/v3_formal_nosmi \
          experiments/v3_formal_noconf experiments/v3_formal_nopadel \
          experiments/v3_formal_ft experiments/v3_formal_ft43 \
          experiments/v3_formal_ft44 \
  --out experiments/v3

# 预测（CSV 必须含 smiles、confseq 和 24 个 PaDEL 列）
molpredict v3-predict \
  --model experiments/v3/runs/V3_full_seed42 \
  --input new_molecules.csv --out predictions.csv

# 测试
./.venv/Scripts/python.exe -m pytest -q
```

所有 run 目录含 `config.json`、`model.pt`、`metrics.json`、`predictions_test.csv`；
`summary.csv`/`stability.json` 汇总三套划分。正式 run 的路径以实际输出目录为准。
