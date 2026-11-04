# molpredict V2 实验报告

## 1. 范围

在保留 M3 CatBoost 基线的同时，实现统一多模态网络：

- ChemBERTa SMILES 预训练编码器（token 级表示）；
- 官方 ConfSeq 预训练编码器（token 级表示，不修改其编码算法）；
- 数值描述符分支：APPf Table S2 的 24 个 PaDEL 描述符，或 RDKit 217 描述符。

三路分支分别投影到 128 维，经可学习注意力池化后进入融合层，最后由共享分类头输出
杀菌剂分类分数。默认冻结两个预训练编码器；额外做 ChemBERTa 最后 2 层部分解冻实验。

架构图与权重来源见 [`v2_architecture.md`](v2_architecture.md)。

### 新增与修改文件

| 文件 | 说明 |
| --- | --- |
| `src/molpredict/padel.py` | APPf Table S2 的 24 个 PaDEL 描述符定义与官方 PaDEL 计算/缓存 |
| `src/molpredict/v2.py` | token 缓存、三套 scaffold 划分、V2Net、两种融合、训练/评估 |
| `scripts/run_v2_experiments.py` | A–G 实验入口、summary/stability 汇总、部分解冻入口 |
| `docs/v2_architecture.md` | 新旧架构图、输入、权重来源、命令、限制 |
| `docs/v2_report.md` | 本报告 |
| `pyproject.toml` | 增加 `v2` extra（padelpy、pandas） |
| `README.md` | 增加 V2 安装/运行/A–G 输入说明 |
| `experiments/v2/` | 缓存、三套划分、每次运行的 config/metrics/model/predictions、summary |
| `experiments/v2/README.md` | 产物目录说明与重建命令 |
| `models/` | 旧 CatBoost 模型原样保留，未删除或覆盖 |

V2 没有修改 `models/` 下的旧模型；A 组是重新训练的同输入 CatBoost 基线，产物在
`experiments/v2/runs/A_catboost_seed*/`。

## 2. 数据与划分

- 数据：`data/training/confseq.csv`，1823 个分子，431 阳性 / 1392 阴性。
- 标签来自 PPDB 农药分类（是否为杀菌剂），**不是实验测定的杀菌活性**。
- Murcko scaffold 分组划分：测试集 20%；剩余训练部分再按 scaffold 划 15% 验证集。
- 三套划分种子 42 / 43 / 44：

| 划分 | 训练 | 验证 | 测试 | 测试阳性 |
| --- | --- | --- | --- | --- |
| seed42 | 1332 (315 阳性) | 168 (36) | 323 (80) | 80 |
| seed43 | 1320 (309 阳性) | 167 (40) | 336 (82) | 82 |
| seed44 | 1295 (285 阳性) | 208 (63) | 320 (83) | 83 |

- 数值描述符的缺失值中位数、均值和标准差**只用训练集拟合**，再应用到验证/测试。
- 分类阈值只在验证集上按最大 F1 选择；测试集只评估一次。

## 3. 预训练权重与加载验证

| 分支 | 来源 | 验证 |
| --- | --- | --- |
| ChemBERTa | `seyonec/ChemBERTa-zinc-base-v1`，Hugging Face 缓存 | `AutoModel`/`AutoTokenizer` 实际加载；44,103,936 参数，hidden 768，6 层，词表 767；缓存 snapshot `761d6a18cf99db371e0b43baf3e2d21b3e865a20`，权重 `pytorch_model.bin` SHA256 `bf8a23d93cca196c3712c94ec379494a74be64c83d86a63b08f03e26db547267` |
| ConfSeq | 官方 `model_epoch_1.pth`（HF 镜像 `Oopstom/confseq-shape-screen`） | 官方 `build_bart_encoder` + `load_checkpoint_to_encoder` 完整加载；14,838,016 参数，hidden 256，6 层；文件 59,396,074 字节，SHA256 `ee1cfa9e391547a6ed3c14876c5331e0029c8608fae67d21181e9fb064599449` |
| PaDEL | padelpy 0.1.17 内置 PaDEL-Descriptor.jar | 24 个名称与 APPf Table S2 逐项匹配，实际计算 1823 行 |

ChemBERTa 权重来自 HF 镜像下载；ConfSeq 官方 OneDrive 链接匿名访问返回 401，因此使用
HF 镜像中的同名 checkpoint。没有用随机初始化冒充预训练模型。

## 4. 实验结果

三套 scaffold 划分上的测试集均值 ± 样本标准差（n=3）。阈值由各划分的验证集单独选择，
测试集只评估一次。

| 实验 | 输入 | 融合 | ROC-AUC | PR-AUC | F1 | Precision | Recall | MCC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | Morgan 2048 + RDKit 217 | CatBoost | **0.8578 ± 0.0292** | **0.6985 ± 0.0285** | **0.6111 ± 0.0513** | 0.5552 ± 0.0283 | 0.6934 ± 0.1414 | **0.4741 ± 0.0566** |
| B | ChemBERTa | — | 0.7579 ± 0.0553 | 0.5299 ± 0.0907 | 0.5344 ± 0.0336 | 0.4546 ± 0.0567 | 0.6769 ± 0.1535 | 0.3586 ± 0.0447 |
| C | ConfSeq | — | 0.7803 ± 0.0193 | 0.5835 ± 0.0294 | 0.5398 ± 0.0427 | 0.5021 ± 0.0501 | 0.5870 ± 0.0628 | 0.3723 ± 0.0595 |
| D | PaDEL 24 | — | 0.6343 ± 0.0260 | 0.3547 ± 0.0021 | 0.4267 ± 0.0256 | 0.3467 ± 0.0141 | 0.5556 ± 0.0515 | 0.1823 ± 0.0390 |
| E | ChemBERTa + ConfSeq | attention | 0.7956 ± 0.0298 | 0.5376 ± 0.0689 | 0.5257 ± 0.0747 | 0.5228 ± 0.0580 | 0.5744 ± 0.2217 | 0.3794 ± 0.0538 |
| E | ChemBERTa + ConfSeq | gated | 0.8044 ± 0.0309 | 0.6251 ± 0.0742 | 0.5563 ± 0.0199 | 0.4510 ± 0.0616 | **0.7589 ± 0.1491** | 0.3896 ± 0.0319 |
| F | ChemBERTa + ConfSeq + PaDEL 24 | attention | 0.7602 ± 0.0062 | 0.5120 ± 0.0887 | 0.5276 ± 0.0026 | 0.4191 ± 0.0230 | 0.7192 ± 0.0715 | 0.3376 ± 0.0096 |
| F | ChemBERTa + ConfSeq + PaDEL 24 | gated | 0.7864 ± 0.0231 | 0.5309 ± 0.0370 | 0.5627 ± 0.0270 | 0.4972 ± 0.0875 | 0.6693 ± 0.0764 | 0.3965 ± 0.0612 |
| G | ChemBERTa + ConfSeq + RDKit 217 | attention | 0.7876 ± 0.0397 | 0.5785 ± 0.0635 | 0.5659 ± 0.0482 | **0.5812 ± 0.1374** | 0.6077 ± 0.1801 | 0.4331 ± 0.0592 |
| G | ChemBERTa + ConfSeq + RDKit 217 | gated | 0.7646 ± 0.0503 | 0.5239 ± 0.0705 | 0.5432 ± 0.0376 | 0.5293 ± 0.0810 | 0.6028 ± 0.1955 | 0.3951 ± 0.0235 |

粗体为该列最佳（只比较同一张表内的 10 个冻结配置）。绝对最佳仍是 CatBoost A；
冻结的 V2 配置中，E/gated 的 AUC、PR-AUC 最高，G/attention 的 F1、MCC 最高，且
G/attention 的 Precision（0.5812）为全表最高。部分解冻的 F-ft 见第 7 节。单模态
PaDEL 24（D）明显弱于其他输入；在 F 中加入 PaDEL 24 没有超过 E。

逐条预测在 `experiments/v2/runs/<实验>_<融合>_seed<N>/predictions_test.csv`，
完整指标在对应 `metrics.json`。

## 5. 多次 scaffold 划分稳定性

下表列出各配置在每套 scaffold 划分上的测试 ROC-AUC。不同种子的测试集有重叠，
因此这不是三次独立复现，只用于观察划分敏感性。

| 配置 | seed42 | seed43 | seed44 | mean ± std |
| --- | --- | --- | --- | --- |
| A CatBoost | 0.8662 | 0.8252 | 0.8819 | 0.8578 ± 0.0292 |
| B ChemBERTa | 0.7369 | 0.7162 | 0.8206 | 0.7579 ± 0.0553 |
| C ConfSeq | 0.7594 | 0.7838 | 0.7976 | 0.7803 ± 0.0193 |
| D PaDEL 24 | 0.6587 | 0.6373 | 0.6069 | 0.6343 ± 0.0260 |
| E attention | 0.8275 | 0.7685 | 0.7907 | 0.7956 ± 0.0298 |
| E gated | 0.7774 | 0.7979 | 0.8380 | 0.8044 ± 0.0309 |
| F attention | 0.7539 | 0.7605 | 0.7664 | 0.7602 ± 0.0062 |
| F gated | 0.7964 | 0.8029 | 0.7600 | 0.7864 ± 0.0231 |
| F-ft（部分解冻） | 0.8349 | 0.7957 | 0.8171 | 0.8159 ± 0.0197 |
| G attention | 0.8332 | 0.7691 | 0.7606 | 0.7876 ± 0.0397 |
| G gated | 0.8226 | 0.7351 | 0.7360 | 0.7646 ± 0.0503 |

观察：

- A（CatBoost）在三套划分上都是最高的均值，且波动不大；这是当前最可靠的配置。
- 神经网络中 F-ft（部分解冻）的均值最高（0.8159），但它只试了一种解冻配置；
  冻结配置中 E/gated 的均值最高（0.8044）。G/attention 在 seed42 达到 0.8332，随后
  两套降到 0.76–0.77，划分敏感性明显。
- C（ConfSeq 单模态）的 AUC 波动最小（0.0193），F/attention 也稳定但均值偏低；
  “稳定”不等于“更好”。
- 单模态 PaDEL 24（D）在三套划分上都明显偏弱，说明这 24 个描述符单独不足以刻画
  当前 PPDB 分类任务。

## 6. 融合方式对比

同一输入、同一划分、同一训练轮数/早停策略下，只改变融合层：

| 输入 | 融合 | AUC | PR-AUC | F1 | MCC |
| --- | --- | --- | --- | --- | --- |
| E | attention | 0.7956 | 0.5376 | 0.5257 | 0.3794 |
| E | gated | **0.8044** | **0.6251** | **0.5563** | **0.3896** |
| F | attention | 0.7602 | 0.5120 | 0.5276 | 0.3376 |
| F | gated | **0.7864** | **0.5309** | **0.5627** | **0.3965** |
| G | attention | **0.7876** | **0.5785** | **0.5659** | **0.4331** |
| G | gated | 0.7646 | 0.5239 | 0.5432 | 0.3951 |

- E（两模态）：gated 的 AUC +0.0088、PR-AUC +0.0875、F1 +0.0306、MCC +0.0102。
- F（三模态，PaDEL）：gated 的 AUC +0.0262、PR-AUC +0.0189、F1 +0.0351、MCC +0.0589。
- G（三模态，RDKit 217）：attention 的 AUC +0.0230、PR-AUC +0.0546、F1 +0.0227、
  MCC +0.0380。

没有一种融合在所有输入上一致占优：gated 在 E/F 更好，attention 在 G 更好。差异大多
在一个标准差以内，n=3，不能据此做显著性结论。一个合理解释是 gated 对模态做全局加权，
在模态数少/数值分支较弱时更稳；attention 允许模态 token 交互，在 217 维 RDKit 分支
信息更丰富时受益。该解释是事后观察，未做消融验证。

阈值方面，F/gated 在 seed42 的验证集最大 F1 阈值低到 0.026，导致高召回、低精确率；
这提醒阈值本身也不稳定，报告 F1/Precision/Recall 时必须同时看阈值。

## 7. 部分解冻微调

对照设置：同一三套 scaffold 划分、同一 F 输入（ChemBERTa + ConfSeq + PaDEL 24）、
同一 attention 融合。只把 ChemBERTa 最后 2 层解冻，编码器学习率 2e-5，其余头部 1e-3，
12 epochs、patience 4；ConfSeq 仍冻结。seed42/43/44 各跑一组。

| 配置（三套均值） | ROC-AUC | PR-AUC | F1 | Precision | Recall | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| F/attention，冻结 | 0.7602 ± 0.0062 | 0.5120 ± 0.0887 | 0.5276 ± 0.0026 | 0.4191 ± 0.0230 | 0.7192 ± 0.0715 | 0.3376 ± 0.0096 |
| F-ft，最后 2 层解冻 | **0.8159 ± 0.0197** | **0.6053 ± 0.0078** | **0.5805 ± 0.0455** | **0.5214 ± 0.0556** | 0.6577 ± 0.0501 | **0.4237 ± 0.0665** |

逐 seed：

| seed | AUC | PR-AUC | F1 | Precision | Recall | MCC | 阈值 | TN/FP/FN/TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | 0.8349 | 0.6049 | 0.6328 | 0.5773 | 0.7000 | 0.5003 | 0.7110 | 202/41/24/56 |
| 43 | 0.7957 | 0.6132 | 0.5500 | 0.4661 | 0.6707 | 0.3804 | 0.6762 | 191/63/27/55 |
| 44 | 0.8171 | 0.5976 | 0.5587 | 0.5208 | 0.6024 | 0.3905 | 0.5288 | 191/46/33/50 |

三套均值上，解冻相对冻结版提升 AUC +0.0557、PR-AUC +0.0933、F1 +0.0529、
MCC +0.0861。seed42 的混淆矩阵从 151/92/18/62 变为 202/41/24/56，假阳性明显减少。

但同一划分的 A 仍是更高：seed42 A 为 0.8662/0.6964/0.6364/0.5057，seed43 为
0.8252/0.6712/0.5521/0.4087，seed44 为 0.8819/0.7280/0.6449/0.5078（AUC/PR-AUC/
F1/MCC，顺序同上）。F-ft 缩小了冻结神经网络的差距，但没有超过 A；且 F-ft 的
Precision/Recall 受阈值影响，seed42 阈值高达 0.711，seed44 为 0.529。

## 8. 混淆矩阵

测试集规模在 320–336 之间，下表给出三套划分的**平均**混淆矩阵，以及 seed42（三套配置
和 A 共用同一测试集，323 个分子）的精确计数。逐条预测在每个 run 目录的
`predictions_test.csv`。

| 配置 | 划分 | TN | FP | FN | TP |
| --- | --- | --- | --- | --- | --- |
| A CatBoost | seed42 | 203 | 40 | 24 | 56 |
| A CatBoost | 三套均值 | 198.7 | 46.0 | 25.0 | 56.7 |
| E gated | seed42 | 151 | 92 | 17 | 63 |
| E gated | 三套均值 | 165.7 | 79.0 | 19.7 | 62.0 |
| F gated | seed42 | 183 | 60 | 25 | 55 |
| F gated | 三套均值 | 185.7 | 59.0 | 27.0 | 54.7 |
| G attention | seed42 | 209 | 34 | 29 | 51 |
| G attention | 三套均值 | 201.0 | 43.7 | 32.0 | 49.7 |
| F-ft（部分解冻） | seed42 | 202 | 41 | 24 | 56 |
| F-ft（部分解冻） | 三套均值 | 194.7 | 50.0 | 28.0 | 53.7 |

G/attention 的假阳性最少、假阴性偏多，偏向保守；E/gated 的假阳性多、假阴性少，偏向
高召回。A 在两者之间取得更好的平衡。

## 9. 与旧 CatBoost 基线的关系

旧 M3 的 `combined` CatBoost（Morgan 2048 + RDKit 217）在 80/20 scaffold 划分上的测试集
指标为 ROC-AUC 0.8778、PR-AUC 0.7307、F1 0.6918；当时早停使用训练集内随机 15% 验证，
阈值固定 0.5。

V2 的 A 组使用**同一测试集**（seed42 的 323 个分子、80 阳性），但训练集为 1332，
另外 168 个分子作为 scaffold 验证集用于早停和选阈值：seed42 为 AUC 0.8662、
PR-AUC 0.6964、F1 0.6364；三套 scaffold 划分均值 AUC 0.8578 ± 0.0292、
PR-AUC 0.6985、F1 0.6111、MCC 0.4741。数值低于旧 80/20 结果，主要因为验证集按
scaffold 划分更严格、训练样本更少，且阈值改由验证集选择；两套数字不能直接混用。
已核对旧 `models/predictions_combined.csv` 与 V2 seed42 测试集的 SMILES：323/323 完全
相同，因此 A 与旧 combined 的差异不是测试集变化造成的。

与 V2 神经网络相比：A 的 AUC（0.8578）和 PR-AUC（0.6985）高于所有神经配置；神经配置
中 F-ft 的 AUC 最高（0.8159），E/gated 的 PR-AUC 最高（0.6251），G/attention 的 MCC
最高（0.4331），但都低于 A（0.4741）。结论是：在当前 1823 个分子、431 个阳性的规模下，
Morgan + RDKit 的 CatBoost 基线仍是首选；V2 的统一多模态网络（含部分解冻）没有在
三套划分上稳定超过基线，暂不应替换。

## 10. 运行命令

```bash
cd D:/works/yuce

# 安装（padelpy 调用 Java；本机 Java 21）
pip install -e ".[confseq,v2]"

# 特征缓存：ChemBERTa/ConfSeq token states、PaDEL 24、RDKit 217
./.venv/Scripts/python.exe scripts/run_v2_experiments.py --prepare

# A–G，三套 scaffold 划分，E/F/G 比较 gated 与 attention
./.venv/Scripts/python.exe scripts/run_v2_experiments.py \
  --experiments A,B,C,D,E,F,G --seeds 42,43,44 \
  --fusions attention,gated --epochs 60 --patience 12

# ChemBERTa 最后 2 层部分解冻（F，seed 42；把 --seeds 改为 43/44 可重复其他划分，
# write_summary 会与已有 summary 合并，不会覆盖 A–G）
./.venv/Scripts/python.exe scripts/run_v2_experiments.py \
  --experiments F --seeds 42 --fusions attention --finetune
```

环境：Python 3.10.11、PyTorch 2.14.1+cpu、Transformers 5.20.2、RDKit 2026.09.1、
scikit-learn 1.7.2、CatBoost 1.2.10、PaDEL-Descriptor (padelpy 0.1.17)、Java 21。
机器有 RTX 4060 8GB，但未安装 CUDA 版 PyTorch；全部实验在 CPU 上完成。

## 11. 未解决问题与限制

- 标签是 PPDB 的农药分类（是否为杀菌剂），不是实验测定的杀菌活性；阴性表示被归为
  除草剂/杀虫剂，不是“实验证明无杀菌活性”。
- APPf 原文的 489/6409 原始数据未公开，本项目用 PPDB 重建 1823 个分子；规模、类别分布
  与论文不同，指标不能与论文直接对比。
- 仅 431 个阳性、三套 scaffold 划分；不同种子的测试集有重叠，mean±std 只反映划分与
  训练随机性，不是独立重复实验，也未做显著性检验。
- PaDEL 24 中，11 个超大/聚合分子被跳过，11 个分子的部分 E-State 为空，`VR1_DzZ` 对
  58 个分子未定义；D/F/G 只用训练集中位数填补，可能影响这些分子的表现。
- 预训练编码器默认冻结；只对 ChemBERTa 最后 2 层在 seed42/43/44 上做了部分解冻，
  未做全量微调，也未解冻 ConfSeq；解冻层数、学习率未调。
- CPU 训练，冻结实验超参数固定（128 维、lr 1e-3、batch 32、60 epochs、patience 12），
  F-ft 为 12 epochs、patience 4，未做系统调参；验证集较小，按最大 F1 选阈值本身有波动。
- 预训练语料与 PPDB 分子可能存在重叠（ChemBERTa 在 ZINC 上预训练，ConfSeq 权重来源
  语料未逐条审计）；本项目只保证训练/验证/测试标签与 scaffold 隔离，未做预训练语料
  去重，因此指标可能偏乐观。这是当前最重要的未验证风险。
- ConfSeq 官方 OneDrive 链接匿名访问 401，checkpoint 取自 HF 镜像；已用官方代码
  `strict=True` 验证加载，但权重再分发/许可证需自行确认（ConfSeq AGPL-3.0）。
- 尚未提供 V2 的独立预测 CLI/外部验证；现有外部验证结果来自旧 CatBoost 模型。
- 本报告中的概率均为模型输出，不是实验活性概率。

## 12. 结论与建议

1. **当前部署仍应使用 A（Morgan + RDKit 217，CatBoost）**：三套 scaffold 划分的均值
   AUC 0.8578、PR-AUC 0.6985、F1 0.6111、MCC 0.4741，AUC/PR-AUC 高于所有 V2 神经
   配置。即使在同一 seed42 划分上，A 的 AUC 0.8662、PR-AUC 0.6964、F1 0.6364、
   MCC 0.5057 也都略高于 F-ft 的 0.8349/0.6049/0.6328/0.5003；F-ft 缩小了差距，
   但没有超过同一划分的 A。
2. V2 的统一架构、三模态输入和两种融合都已跑通并可复现；神经网络中 F-ft 的 AUC
   （0.8159）和 F1（0.5805）最高，E/gated 的 PR-AUC（0.6251）最高，G/attention 的
   MCC（0.4331）最高。
3. PaDEL 24 单独使用（D）明显弱；在 F 中加入它没有稳定超过 E。APPf 的 24 个描述符
   更像“论文特定特征”，不能假设在 PPDB 重建数据上同样有效。
4. 部分解冻是缩小与 A 差距最明显的 V2 变体：三套均值 AUC 0.8159、PR-AUC 0.6053、
   F1 0.5805、MCC 0.4237，仍低于 A（0.8578/0.6985/0.6111/0.4741）。下一步应优先
   扩大微调实验（多解冻层数/学习率/融合方式），而不是继续堆叠冻结特征。
5. 下一步优先级：审计预训练语料重叠；增加阳性样本/外部实验数据；在验证集上做
   超参数与融合方式的嵌套选择；为 V2 增加独立预测 CLI 和外部验证。否则不应把 V2
   当作比 CatBoost 更可靠的筛选工具。
