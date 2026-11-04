# molpredict V3：统一多模态 Transformer

## 1. 目标

V3 不是三个独立模型，也不是先各自池化再拼三个向量。它把同一个分子的三种信息放进
**一条 token 序列**，在同一个多层 Transformer 中共同计算注意力，最后只输出一个
PPDB 杀菌剂分类分数。

```text
SMILES 字符串 ─ ChemBERTa tokenizer ─ ChemBERTa 编码器（默认冻结）
               → token states [L_s, 768] ─ 投影 ─┐
                                                  │
ConfSeq 字符串 ─ 官方 ConfSeq 词表/编码器（默认冻结）│
               → token states [L_c, 256] ─ 投影 ─┤
                                                  ├─ 统一 token 序列
24 项 PaDEL ─ 24 个独立数值 token ─ 数值 MLP ─────┘
               + 特征身份 embedding

[CLS] + SMILES tokens + ConfSeq tokens + 24 descriptor tokens
                         │
        共同的 4 层 Transformer（同一注意力矩阵）
                         │
                  CLS 表示 → 分类头 → 一个分数
```

## 2. 三种输入

### 2.1 SMILES

- 预训练模型：`seyonec/ChemBERTa-zinc-base-v1`。
- 使用其原始 tokenizer，`max_length=320`；V3 缓存实测最大 token 长度 298，**未截断**。
  V2 的 128 截断只影响 8 个长分子，V3 重新生成了完整缓存。
- 保留 token 级 `last_hidden_state`，不提前池化；保留 attention mask。
- ChemBERTa 自己的位置编码和结构没有被移除；进入融合网络前再叠加可学习的模态内
  位置 embedding 和模态 embedding。

### 2.2 ConfSeq

- 官方实现和官方权重：`data/external/ConfSeq/representation_learning` 的
  `build_bart_encoder` / `load_checkpoint_to_encoder`，checkpoint 为
  `model_epoch_1.pth`（HF 镜像 `Oopstom/confseq-shape-screen`）。
- 官方 ConfSeq 输入是空格分隔的 token；按官方词表查 id，pad 使用官方 pad id。
- 保留 token 级隐藏表示 `[L_c, 256]`，不提前池化；最大长度 512（当前数据最大 499）。
- 没有用 RDKit 3D 坐标替代 ConfSeq，也没有改官方模型算法。

### 2.3 24 项 PaDEL

严格使用 APPf 补充材料 Table S2 的 24 个名称和 PaDEL-Descriptor 计算，不使用 RDKit
相似特征替代。顺序固定：

```text
nBr, AATS7m, ATSC0c, AATSC5m, AATSC6m, VR1_DzZ,
nHdNH, nHsSH, nHsNH2, nHsssNHp, nHCsatu, nssssNp,
naaO, nssssSi, nddssS, SsBr, minHBint8, minHBint9,
n7Ring, nG12Ring, nF6Ring, nF7Ring, nF8Ring, n6HeteroRing
```

每项特征变成一个独立 token：

```text
token_i = MLP(标准化的第 i 个数值) + feature_identity_embedding[i] + modality_embedding
```

24 项数值**不是**原子位置序列。V3 不假设第 i 个描述符对应 SMILES 或 ConfSeq 的第 i 个
token；它们只通过共同注意力学习统计关联。

标准化参数和缺失值中位数**只在训练集拟合**；验证/测试使用训练集参数。缺失值记录在
`experiments/v2/cache/padel24_meta.json`，特征名和顺序保存在 checkpoint 的
`feature_order.json`。

## 3. 统一融合

### 3.1 共同 token 序列

```text
[CLS] + SMILES tokens + ConfSeq tokens + 24 descriptor tokens
```

- 每个 token 有：
  - 来自各自编码器的内容表示（SMILES/ConfSeq）或数值+身份表示（descriptor）；
  - 模态 embedding（smiles / confseq / descriptor）；
  - SMILES/ConfSeq 的可学习模态内位置 embedding；
  - descriptor 的特征身份 embedding。
- 使用 `src_key_padding_mask` 屏蔽 padding；无效位置不参与注意力。
- 融合网络是 4 层、4 头、128 维、FFN 256 的 Transformer Encoder。
- 同一个注意力矩阵中，SMILES token 可以看 ConfSeq/descriptor，ConfSeq token 可以看
  SMILES/descriptor，descriptor token 之间及与两种序列也可以互相参考。
- 最终取 `[CLS]` 输出，经 `LayerNorm → Linear(128→64) → GELU → Dropout → Linear(64→1)`
  得到一个 logit，训练用 `BCEWithLogitsLoss`。

### 3.2 为什么不是 V2 的做法

V2 的 E/F/G 先对每个分支做注意力池化得到 `z_s/z_c/z_n`，再在三个摘要向量上做门控或
注意力融合。V3 保留这些历史实验作为对照，但 V3 自身在池化之前就把三种 token 序列
放进共同 Transformer；融合发生在 token 级，而不是三个摘要之间。

## 4. 训练策略

第一阶段（主实验）：

- ChemBERTa 和 ConfSeq 都冻结，只使用缓存 token states；
- 训练投影层、模态/位置 embedding、24 个 descriptor token 编码、4 层融合 Transformer
  和分类头；
- 分类损失统一回传到所有可训练参数。

第二阶段（可选）：

- `full_ft`：把 ChemBERTa 最后 2 层解冻，编码器学习率 `2e-5`；融合网络学习率与
  主实验一致（本次正式对照使用 `3e-4`、batch 32、12 epochs、patience 4）；
- ConfSeq 仍冻结；是否解冻 ConfSeq 由验证结果决定，不能为了“端到端”而全量解冻。
- 由于 431 个阳性样本，默认不做两个编码器的全量微调。

## 5. 数据划分与隔离

- 数据：`data/training/confseq.csv`，1823 个分子（431 阳性 / 1392 阴性）。
- 复用 V2 的 Murcko scaffold 三套划分（seed 42/43/44），坐标复制到
  `experiments/v3/splits/`，训练/验证/测试索引严格一致。
- 测试集只做最终评估；早停和阈值都只使用验证集。
- 标准化和中位数填补只用训练集拟合。
- `scripts/run_v3_experiments.py` 同时生成 `experiments/v3/leakage_audit.json`：
  InChIKey 重复、canonical SMILES 重复、split 间索引/InChIKey/Murcko scaffold 重叠。
  预训练语料重叠只能标记为 `unknown`，因为没有完整 ZINC/ConfSeq 预训练分子清单。

## 6. Checkpoint 与预测

每个 run 目录保存：

```text
model.pt                 # 融合网络（live 实验还含解冻编码器）
config.json              # 结构、训练配置、阈值、数据/缓存哈希、预训练来源
tokenizer/               # ChemBERTa tokenizer（含 smiles 模态时）
confseq_vocab.json       # ConfSeq 词表、pad/unk、官方 checkpoint 哈希
standardization.json     # 训练集 median/mean/std
feature_order.json       # 24 项 PaDEL 的固定顺序
split.json               # 本 run 使用的 train/val/test 索引
pretrained.json          # 预训练来源、哈希、许可证状态
metrics.json             # val/test/history
predictions_test.csv     # 测试集逐条预测
```

预测入口：

```bash
molpredict v3-predict \
  --model experiments/v3/runs/V3_full_seed42 \
  --input new_molecules.csv \
  --out predictions.csv
```

输入 CSV 必须包含 `smiles`、`confseq` 和 24 个 PaDEL 列；可以额外包含 `name`/`label`。
ConfSeq 必须由官方 ConfSeq 流程生成，PaDEL 必须由官方 PaDEL-Descriptor 生成；预测接口
不会用 RDKit 3D 伪造 ConfSeq，也不会自行替换 24 项描述符。

输出分数是 PPDB 杀菌剂分类模型输出，**不是**实验抗真菌活性概率。

## 7. 许可证与来源（当前状态）

| 组件 | 来源 | 许可证状态 |
| --- | --- | --- |
| ChemBERTa 代码 | 上游仓库 | MIT（代码）；权重许可证未在本地 model card 单独确认 |
| ChemBERTa 权重 | `seyonec/ChemBERTa-zinc-base-v1` | 未确认，非商业研究使用 |
| ConfSeq 代码/权重 | 官方仓库 + HF 镜像 checkpoint | AGPL-3.0；再分发需确认 |
| PaDEL-Descriptor | padelpy 0.1.17 内置 jar | AGPL-3.0 |
| RDKit | 项目依赖 | BSD-3-Clause |
| PyTorch / Transformers / scikit-learn / CatBoost | 项目依赖 | 各自开源许可证，使用前按发行版确认 |

本项目仅用于学习和非商业研究；未确认的许可证不能声称可自由商用。

## 8. 已知限制

- 标签是 PPDB 农药分类，不是实验活性。
- 431 个阳性，模型不确定性大；三套划分测试集有重叠。
- PaDEL 24 的 11 个超大分子缺失、11 个分子部分 E-State 缺失；训练集中位数填补。
- 预训练语料与 PPDB 可能重叠，目前无法逐分子审计。
- CPU 训练；`full_ft` 只解冻 ChemBERTa 最后 2 层。
- 注意力权重不能解释为真实化学作用位点。

## 9. 与 V2 的设计差异

| 项目 | V2 | V3 |
| --- | --- | --- |
| 融合位置 | 三个池化向量之后 | token 级共同 Transformer |
| SMILES/ConfSeq | 先注意力池化 | 保留 token 序列，与 descriptor 同序列 |
| 数值特征 | 一个向量分支 | 24 个独立 token + 特征身份 |
| 位置信息 | 编码器内部 + 分支池化 | 编码器内部 + 模态内可学习位置 + 模态 embedding |
| 预训练微调 | F-ft 最后 2 层 | 默认冻结；`full_ft` 最后 2 层对照 |
| 对照 | A–G | V2 A/F/F-ft + V3 全量/三组消融 |
