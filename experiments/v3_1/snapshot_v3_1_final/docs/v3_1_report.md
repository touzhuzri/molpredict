# molpredict V3.1 报告：GPU、加深融合与学习行为审计

> 本报告只讨论 PPDB 农药分类标签上的模型指标，不是实验抗真菌活性。
> V3 的原始结果、代码快照和权重均保留在 `experiments/v3/`；V3.1 写入
> `experiments/v3_1/`，两者不互相覆盖。

## 0. 执行摘要

- 当前 PyTorch 原为 **CPU-only**（`2.14.1+cpu`，`torch.cuda.is_available()=False`），
  即使机器上有 RTX 4060 Laptop。V3.1 安装了 `torch==2.14.1+cu130` 并实际验证了
  CUDA、BF16、ChemBERTa/ConfSeq/融合网络的设备与反向传播。
- 对 V3 的审计发现：**不是容量不足，而是严重过拟合与模态不平衡**。V3 `full`
  seed42 的训练 AUC 为 0.9963，测试 AUC 只有 0.6854（差 0.311）；去掉 SMILES 后
  训练/测试差距从 0.311 降到 0.115。SMILES 分支的梯度很大、tokenizer 与 mask 正常、
  没有截断，所以问题不是“SMILES 没接上”，而是它在小数据上拟合了训练骨架。
- 加深/加宽没有带来可复现的收益：验证 AUC 为 base 0.8345、L 0.8368、
  M 0.8203、D 0.8385，差异小于 seed 间波动；测试 AUC 为 base 0.7718、
  L 0.7852、M 0.7813、D 0.7639。容量不是主要瓶颈。
- 部分微调才是有效改动：`ft_smiles` 0.8265 ± 0.0287、`ft_both`
  0.8342 ± 0.0307，均高于冻结 base（0.7718）和 V3 full_ft（0.8191），
  但仍低于 CatBoost A（0.8578）。
- 冻结消融：`no_smiles` 0.8034（+0.0316）、`no_confseq` 0.7284（−0.0434）、
  `no_padel` 0.7797（+0.0079）。ConfSeq 是冻结设置下最关键的模态。
- 补充的 `ft_no_smiles`（微调 ConfSeq、去掉 SMILES）为 0.8003；保留 SMILES 的
  `ft_smiles`/`ft_both` 更高，说明 SMILES 在适配后是正贡献，V3 的“SMILES 拖累”
  主要来自冻结的 ChemBERTa 表示。
- 建议：保持 128/4/4；没有证据支持 192/256 或 8 层；若追求精度，部分微调
  ChemBERTa（可再加 ConfSeq）是唯一稳定的增益，但代价是 2.3–3.4 倍训练时间。
  当前主要瓶颈是数据规模、scaffold 外推与验证集噪声，不是模型容量。
- 补充外部验证（21 个未参与训练的已知分子，8 阳性）：CatBoost AUC 0.750，
  V3.1 最好的 `ft_no_smiles` 0.673、`ft_both` 0.558；在 10 个 scaffold-disjoint
  分子上所有模型都接近/低于随机。**内部测试提升没有转化为外部泛化。**

## 1. 保护与基线

执行前检查并等待了正在运行的 V3 `full_ft` seed43/44 结束，然后：

- 汇总 15 个 V3 run 到 `experiments/v3/summary.csv` / `stability.json`；
  `full_ft` 三划分 AUC = 0.8191 ± 0.0210。
- 将 V3 代码、文档、汇总复制到 `experiments/v3/snapshot_v3_final/`，并写入
  `SHA256SUMS.json`。
- 为注意力审计给 `V3Net` 增加了只读的 `embed()` 方法；`forward()` 改为调用它。
  该改动不新增参数、不改变计算图，审计复算的测试 AUC 与 V3 已保存指标完全一致
  （0.6854423868）。
- 所有 V3.1 输出使用独立目录；V2/V3 的权重与报告未被覆盖。

V3 基线（同一数据、同一 scaffold 划分、测试集只评估一次）：

| 配置 | n | 测试 AUC | 测试 PR-AUC | 测试 F1 | 测试 MCC |
| --- | --- | --- | --- | --- | --- |
| V3 `full` | 3 | 0.7513 ± 0.0571 | 0.5014 ± 0.0589 | 0.4689 ± 0.0652 | 0.2981 ± 0.0544 |
| V3 `no_smiles` | 3 | 0.8066 ± 0.0222 | 0.6024 ± 0.0695 | 0.5682 ± 0.0428 | 0.4033 ± 0.0576 |
| V3 `no_confseq` | 3 | 0.7155 ± 0.0749 | 0.4612 ± 0.0474 | 0.4741 ± 0.0723 | 0.2530 ± 0.1034 |
| V3 `no_padel` | 3 | 0.7850 ± 0.0248 | 0.5719 ± 0.0402 | 0.5495 ± 0.0126 | 0.3830 ± 0.0294 |
| V3 `full_ft` | 3 | 0.8191 ± 0.0210 | 0.6223 ± 0.0296 | 0.5825 ± 0.0312 | 0.4240 ± 0.0399 |

对照：V2 A CatBoost 0.8578 ± 0.0292、V2 F 0.7602 ± 0.0062、
V2 F-ft 0.8159 ± 0.0197（AUC）。

## 2. GPU 环境核验

### 2.1 安装前：有 GPU，但 PyTorch 是 CPU 版

| 项目 | 值 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| 显存 | 8188 MiB（8 GB） |
| 驱动 | 616.92；CUDA UMD 13.4 |
| PyTorch | 2.14.1+cpu |
| torch.version.cuda | None |
| torch.cuda.is_available() | False |
| 训练时桌面应用占用 | 约 2.5–3.4 GB |

原因不是驱动或显卡，而是 `.venv` 里装的是 CPU-only wheel。V3 全部实验因此在 CPU
上训练。

### 2.2 安装与实测

安装了 `torch==2.14.1+cu130`（CUDA 13.0 构建，匹配驱动；wheel 经 NJU PyTorch
镜像下载并做 ZIP CRC 校验），然后实际运行：

| 检查 | 结果 |
| --- | --- |
| `torch.__version__` | 2.14.1+cu130 |
| `torch.version.cuda` | 13.0 |
| `torch.cuda.is_available()` | True |
| 设备 | RTX 4060 Laptop，capability 8.9 |
| cuDNN | 92400 |
| BF16 | 支持 |
| 1024×1024 CUDA 矩阵乘法 | 通过 |

`scripts/check_v3_1_gpu.py` 用真实数据构建 ChemBERTa + ConfSeq + 192/6/4 融合网络：

- 所有模块参数都在 `cuda:0`：ChemBERTa 103 个参数张量、ConfSeq 103 个、
  投影层各 5 个、融合层 72 个、分类头 6 个；`all_params_on_device=True`。
- 输入 batch（`smiles_ids`、`confseq_ids`、`descriptors`、`label`）都在 `cuda:0`。
- FP32 前向通过，batch=4 峰值分配显存 250.6 MB。
- BF16 autocast 前向 + 反向通过；峰值分配 343.0 MB、峰值保留 386.0 MB，
  反向耗时 0.283 s（batch=4）。
- 反向梯度非零：`smiles_encoder` 0.146、`confseq_encoder` 0.971、
  `smiles_projector` 0.269、`confseq_projector` 0.495、`descriptor_encoder`
  0.169、`fusion` 1.845、`classifier` 2.569。
- 两个编码器都解冻最后 2 层时：总参数 61,118,977，可训练 21,084,161。

正式训练中每个 epoch 还记录 `nvidia-smi` 的利用率和显存占用（见 `history.json`）；
冻结模型训练的典型利用率约 60%–70%，峰值分配显存约 0.4–1.0 GB，说明瓶颈不是显存
而是小数据与泛化。

### 2.3 数据传输

- SMILES/ConfSeq 缓存和 PaDEL 矩阵在启动时一次性读入 CPU 内存；
- 每个 batch 只传当前配置需要的键（冻结模态传 token states，live 模态传 token ids）；
- DataLoader 使用 `pin_memory` 和 `non_blocking`；
- 冻结编码器不会在每个 batch 重新编码；微调时使用实时编码，不用旧缓存替代。

## 3. V3 学习行为审计

审计对象：`experiments/v3/runs/` 中已保存的正式 checkpoint。审计脚本
`scripts/audit_v3.py`，输出 `experiments/v3_1/audit/`。

### 3.1 参数量

V3 `full`（128/4/4，冻结编码器、缓存 token 表示）：

| 模块 | 参数量 |
| --- | ---: |
| ChemBERTa（外部冻结编码器） | 44,103,936 |
| ConfSeq（外部冻结编码器） | 14,838,016 |
| 融合网络总计 | 789,121 |
| 其中可训练 | 789,121 |
| 其中冻结 | 0 |

融合网络内部（来自 `parameter_breakdown`）：

| 模块组 | 参数量 |
| --- | ---: |
| SMILES 投影 + 模态位置 | 139,648 |
| ConfSeq 投影 + 模态位置 | 98,688 |
| 24 descriptor token 编码器 | 11,520 |
| 融合 Transformer（4 层） | 529,920 |
| 分类头 | 8,577 |
| CLS/模态 embedding/input norm | 768 |

`full_ft` 总参数 44,893,057，可训练 14,964,865；后者的构成是 ChemBERTa 最后 2 层
（14,175,744）+ 融合网络 789,121。ChemBERTa embeddings、pooler 和前 4 层保持冻结。

### 3.2 训练过程

V3 正式 run 的逐 epoch 记录只有 `train_loss` 和 `val_roc_auc`。审计复算了最终
checkpoint 在训练集上的表现：

| 配置/seed | 训练 AUC | 验证 AUC | 测试 AUC | 训练−测试 |
| --- | ---: | ---: | ---: | ---: |
| `full` 42 | 0.9963 | 0.8199 | 0.6854 | 0.3109 |
| `full` 43 | 0.9884 | 0.7780 | 0.7809 | 0.2075 |
| `full` 44 | 0.9884 | 0.8842 | 0.7875 | 0.2009 |
| `no_smiles` 42 | 0.9031 | 0.7864 | 0.7876 | 0.1155 |
| `no_confseq` 42 | 0.9610 | 0.8125 | 0.7461 | 0.2149 |
| `no_padel` 42 | 0.9108 | 0.8053 | 0.7737 | 0.1371 |
| `full_ft` 42 | 0.9850 | 0.8258 | 0.8192 | 0.1657 |

- 训练损失在每个 run 都持续下降（`full` seed42：1.060 → 0.169），没有欠拟合；
  验证损失在后期上升，测试 AUC 与验证 AUC 在 scaffold 外推时脱节。
- 三套划分的差距都很大（0.20–0.31），说明不是单个 seed 的偶发。
- `full` seed42 的最佳 epoch 正好是 20（达到 20 epoch 上限），验证 AUC 仍在上升；
  早停没有阻止对训练骨架的拟合。其它 run 的最佳 epoch 为 3–18，说明“训练轮数不够”
  不是所有配置的共同瓶颈，更常见的问题是验证集上的继续提升不迁移到测试骨架。
- 去掉 SMILES 后差距缩小一半以上，是最强的“SMILES 导致过拟合”证据。

曲线见 `experiments/v3_1/audit/V3_full_seed42_curves.png` 等。

### 3.3 梯度

一个训练 batch 的梯度 L2 范数（seed42；只代表该 checkpoint 的瞬时梯度）：

| 模块组 | `full` | `no_smiles` | `no_confseq` | `no_padel` |
| --- | ---: | ---: | ---: | ---: |
| SMILES 投影 | 0.497 | — | 8.254 | 1.797 |
| ConfSeq 投影 | 0.399 | 4.007 | — | 3.544 |
| descriptor 编码器 | 0.026 | 0.107 | 0.137 | — |
| 融合层 0 | 0.429 | 1.563 | 4.992 | 1.182 |
| 融合层 3 | 0.053 | 0.165 | 0.346 | 0.257 |
| 分类头 | 0.236 | 0.866 | 1.123 | 0.677 |
| 总梯度 | 0.878 | 4.648 | 10.238 | 4.414 |

- 三个输入分支都拿到了非零梯度，没有 detach/缓存切断计算图。
- SMILES/ConfSeq 投影梯度远大于 descriptor 编码器；24 个 descriptor token 的
  学习信号相对弱。
- 融合层梯度随深度快速衰减（第 0 层到第 3 层约 8–14 倍），提示 4 层融合的
  后层在当前设置下学习信号较弱。
- `full_ft`：ChemBERTa 最后 2 层（14,175,744 参数）全部有梯度（范数 3.50），
  embeddings/pooler 无梯度；冻结参数没有被更新。

### 3.4 Token 与注意力

- SMILES 长度：均值 28.1、中位数 25、最大 298、超过 320 的 0 条。
- ConfSeq 长度：均值 55.2、中位数 52、最大 499、超过 512 的 0 条。
- 24 个 descriptor token 全部有效；padding mask 通过测试验证。
- tokenizer 往返：前 5 条 SMILES 解码后与原字符串完全一致，没有分词/拼接错误。
- 注意力每层每头都统计，`attention.json` 保留每一层的 `heads`（按头）和 `mean`
  （头均值）份额；下面只展示头均值。以 `full` 最后一层
  的均值头为例（份额 = 源 token 到该目标组的注意力比例；其余主要给 CLS）：

| 源 → 目标 | SMILES | ConfSeq | descriptors |
| --- | ---: | ---: | ---: |
| SMILES | 0.071 | 0.121 | 0.059 |
| ConfSeq | 0.077 | 0.120 | 0.053 |
| descriptors | 0.072 | 0.121 | 0.058 |

- 存在真实的跨模态注意力：SMILES→ConfSeq 0.121、ConfSeq→SMILES 0.077、
  descriptors→SMILES 0.072、descriptors→ConfSeq 0.121（最后一层）。
- 但相当大比例的注意力流向 CLS（约 0.75），且各模态 token 对彼此的份额
  只有 5%–12%；不能据此声称模型发现了化学作用位点。

### 3.5 数据与标签

- 1823 个分子，431 阳性 / 1392 阴性。阳性 = PPDB fungicide 列表；阴性 = PPDB
  herbicide 或 insecticide 列表。**这不是实验抗真菌活性**，阴性也不等于实验无活性。
- scaffold 划分（seed 42/43/44）训练/验证/测试分别为 1332/168/323、
  1320/167/336、1295/208/320。
- 24 个 PaDEL 列全部存在 NaN，列缺失数 11–58；V3/V3.1 只在训练集上拟合中位数
  填补与标准化，再变换验证/测试。
- 泄漏审计：seed 42/43/44 的 InChIKey 重复 0、canonical SMILES 重复 0；
  train/val/test 的索引、InChIKey、Murcko scaffold 重叠均为 0。
- 预训练语料与当前数据的重叠仍为 **unknown**：没有 ZINC/ChemBERTa 和 ConfSeq
  预训练语料的完整分子清单，不能声称不存在重叠。
- SMILES 与 ConfSeq 的对应关系由同一 1823 行数据保证，缓存逐行校验 InChIKey；
  PaDEL 列顺序由 `padel24_order.json` 固定。


## 4. V3.1 实现

V3.1 **不改变** V3 的统一架构：仍是
`[CLS] + SMILES token states + ConfSeq token states + 24 descriptor tokens`
进入同一个 `nn.TransformerEncoder`，CLS 接一个分类头。改动集中在训练配方、
可选的深度/宽度和诊断记录。

### 4.1 新增代码

| 文件 | 说明 |
| --- | --- |
| `src/molpredict/v3_1_train.py` | GPU/AMP 训练、warmup+cosine、梯度累积、逐轮诊断、注意力审计 |
| `scripts/run_v3_1_experiments.py` | 结构 / 微调 / 消融实验入口 |
| `scripts/check_v3_1_gpu.py` | 实际 CUDA/模块设备/AMP/反向传播核验 |
| `scripts/audit_v3.py` | V3/V3.1 checkpoint 的学习行为审计 |
| `tests/test_v3_1.py` | embed 布局、mask、注意力归一化、梯度组、调度器、结构整除性 |

### 4.2 结构配置

| 配置 | d_model | 层数 | 头数 | FFN | 融合网络参数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `base`（V3 结构） | 128 | 4 | 4 | 256 | 789,121 |
| `L` | 192 | 6 | 4 | 384 | 2,177,025 |
| `M` | 256 | 6 | 8 | 512 | 3,697,281 |
| `D` | 128 | 8 | 4 | 256 | 1,319,041 |

所有配置都保持相同的三种输入、共同注意力、CLS 分类、标签和 scaffold 划分；
`d_model` 均能被头数整除。

### 4.3 训练配方

```text
优化器      AdamW(weight_decay=1e-4)
学习率      融合网络 3e-4；部分解冻编码器 2e-5
调度        warmup 10% + cosine 衰减到 0.1×基础 lr
batch       16 × 梯度累积 2 = 有效 32
epochs      最多 100；验证 ROC-AUC 早停 patience=18
精度        BF16 autocast（GPU 支持时）；损失在 FP32 计算
裁剪        grad norm 5.0
阈值        验证集最大 F1；测试集只评估一次
```

- 学习率分两组：预训练编码器 2e-5；模态投影、descriptor 编码器、融合层和
  分类头共用融合 lr 3e-4。没有为每个子模块单独设 lr（当前数据量下没有依据支持
  更细的分组），但所有组都记录了实际 lr。
- 每个 epoch 记录：训练/验证 loss、训练/验证 AUC 与 PR-AUC、各组 lr、
  按模块组的梯度范数、epoch 耗时、GPU 峰值显存、`nvidia-smi` 利用率/占用。
- 每个 run 结束保存 `history.json`、`attention.json`（按层/按头）、
  `metrics.json`、`config.json`（含参数量、设备、代码哈希、数据哈希）和权重。
- 数据侧：缓存只在启动时从 CPU 读入；每个 batch 只把当前配置需要的张量
  传到 GPU；冻结编码器使用缓存，live 编码器不使用旧缓存。

## 5. 实验设计

### 5.1 分组

| 组 | 实验 | 结构 | 编码器状态 | 目的 |
| --- | --- | --- | --- | --- |
| 1 | `base` | 128/4/4 | 两个都冻结（缓存） | V3 结构在新配方下的基线 |
| 1 | `L` | 192/6/4 | 冻结 | 增加宽度与深度 |
| 1 | `M` | 256/6/8 | 冻结 | 最大容量（显存允许时） |
| 1 | `D` | 128/8/4 | 冻结 | 只加深 |
| 2 | `ft_smiles` | 组 1 选定 | ChemBERTa 最后 2 层解冻，ConfSeq 冻结 | 部分微调 SMILES 编码器 |
| 2 | `ft_both` | 组 1 选定 | ChemBERTa 与 ConfSeq 各解冻最后 2 层 | 两个编码器有限微调 |
| 3 | `no_smiles` | 组 1 选定 | 冻结 | 三模态消融 |
| 3 | `no_confseq` | 组 1 选定 | 冻结 | 三模态消融 |
| 3 | `no_padel` | 组 1 选定 | 冻结 | 三模态消融 |

每组 3 个 scaffold 划分（seed 42/43/44）。对照组直接复用 V2 A / F / F-ft 和
V3 原始结果，不重复训练。

### 5.2 选择规则

- 结构选择只看**验证集**平均 ROC-AUC、跨 seed 标准差和计算成本，不按测试集挑结构。
- epoch 与阈值只由验证集决定；测试集在每个 run 内只评估一次。
- 组 2、组 3 使用组 1 选定的结构，避免把结构变化与微调/消融混在一起。
- 不使用测试集回选结构、学习率、轮数或阈值。

### 5.3 计算资源

- GPU：NVIDIA GeForce RTX 4060 Laptop，8 GB；测试时桌面应用已占用约 2.7 GB，
  可用约 5.4 GB。
- 冻结实验只训练融合网络（0.79–3.70 M 参数），token 表示来自缓存。
- `ft_smiles` 训练 ChemBERTa 最后 2 层 + 融合网络；`ft_both` 再训练 ConfSeq
  最后 2 层（4.73 M）。不会用缓存替代正在微调的编码器。
- 显存不足时优先降低 batch 并提高梯度累积，保持有效 batch 不变；不丢 token。

实测（同一 8 GB GPU，桌面应用占用约 2.5–4 GB）：

| 配置 | 每 epoch 秒 | 峰值分配显存 |
| --- | ---: | ---: |
| base（冻结） | 1.41 | 429 MB |
| L（冻结） | 2.32 | 659 MB |
| M（冻结） | 2.63 | 873 MB |
| D（冻结） | 2.45 | 575 MB |
| ft_smiles | 3.20 | 991 MB |
| ft_both | 4.76 | 1475 MB |

训练时 `nvidia-smi` 利用率约 60%–86%，温度 72–76°C；没有发生 OOM，也没有退回 CPU。

## 6. 结构实验（第一组）

所有结构使用同一缓存、同一 scaffold 划分、同一训练配方（BF16、batch 16×2、
lr 3e-4、dropout 0.3、最多 100 epoch、patience 18），只改变 d_model/层数/头数。

| 实验 | 结构 | n | 测试 AUC | 测试 PR-AUC | 测试 F1 | 测试 MCC | 验证 AUC | 训练 AUC | 训练-测试 | 最佳 epoch | 参数量 | 每 epoch 秒 | 峰值显存 MB |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| `D` | 128/8/4 | 3 | 0.7639 ± 0.0361 | 0.5528 ± 0.0144 | 0.5306 ± 0.0295 | 0.3518 ± 0.0267 | 0.8385 ± 0.0297 | 0.9883 ± 0.0134 | 0.2244 ± 0.0443 | 30/25/16 | 1,319,041 | 2.45 | 575 |
| `L` | 192/6/4 | 3 | 0.7852 ± 0.0115 | 0.5898 ± 0.0420 | 0.5353 ± 0.0210 | 0.3582 ± 0.0450 | 0.8368 ± 0.0490 | 0.9768 ± 0.0249 | 0.1917 ± 0.0240 | 38/11/16 | 2,177,025 | 2.32 | 659 |
| `M` | 256/6/8 | 3 | 0.7813 ± 0.0097 | 0.5727 ± 0.0355 | 0.5456 ± 0.0154 | 0.3763 ± 0.0251 | 0.8203 ± 0.0423 | 0.9630 ± 0.0179 | 0.1818 ± 0.0160 | 12/11/13 | 3,697,281 | 2.63 | 873 |
| `base` | 128/4/4 | 3 | 0.7718 ± 0.0312 | 0.5324 ± 0.0646 | 0.5418 ± 0.0322 | 0.3930 ± 0.0190 | 0.8345 ± 0.0429 | 0.9948 ± 0.0025 | 0.2230 ± 0.0291 | 20/23/26 | 789,121 | 1.41 | 429 |

逐 seed：

| 实验 | seed | 验证 AUC | 测试 AUC | PR-AUC | F1 | MCC | 最佳 epoch | 运行 epoch | 训练 AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `base` | 42 | 0.8453 | 0.7528 | 0.5230 | 0.5464 | 0.3717 | 20 | 38 | 0.9922 |
| `base` | 43 | 0.7872 | 0.8078 | 0.6012 | 0.5075 | 0.4083 | 23 | 41 | 0.9972 |
| `base` | 44 | 0.8709 | 0.7548 | 0.4729 | 0.5714 | 0.3990 | 26 | 44 | 0.9949 |
| `L` | 42 | 0.8245 | 0.7807 | 0.5676 | 0.5595 | 0.4060 | 38 | 56 | 0.9996 |
| `L` | 43 | 0.7951 | 0.7766 | 0.5636 | 0.5222 | 0.3519 | 11 | 29 | 0.9503 |
| `L` | 44 | 0.8907 | 0.7982 | 0.6382 | 0.5242 | 0.3168 | 16 | 34 | 0.9806 |
| `M` | 42 | 0.8224 | 0.7906 | 0.5823 | 0.5634 | 0.3943 | 12 | 30 | 0.9607 |
| `M` | 43 | 0.7770 | 0.7712 | 0.6024 | 0.5366 | 0.3870 | 11 | 29 | 0.9464 |
| `M` | 44 | 0.8615 | 0.7821 | 0.5334 | 0.5368 | 0.3477 | 13 | 31 | 0.9820 |
| `D` | 42 | 0.8485 | 0.7222 | 0.5375 | 0.5225 | 0.3299 | 30 | 48 | 0.9962 |
| `D` | 43 | 0.8051 | 0.7857 | 0.5659 | 0.5060 | 0.3440 | 25 | 43 | 0.9958 |
| `D` | 44 | 0.8619 | 0.7837 | 0.5551 | 0.5634 | 0.3815 | 16 | 34 | 0.9727 |

### 6.1 结构加深/加宽是否有效？

- 验证 AUC 均值：D 0.8385、L 0.8368、base 0.8345、M 0.8203；四者差异
  （最大 0.018）小于 seed 间标准差（0.03–0.05），没有统计上可分辨的排序。
- 测试 AUC 均值：L 0.7852、M 0.7813、base 0.7718、D 0.7639。L 看似最好，但这是
  事后观察，不能用于回选结构；验证集并不支持“L 明显更好”。
- 所有结构训练 AUC 仍为 0.963–0.995，训练-测试差 0.18–0.22；增加容量没有消除
  过拟合，M 的训练 AUC 反而最低（0.963），说明它并没有更好地拟合训练集。
- 计算成本：base 1.41 s/epoch、峰值 429 MB；L 2.32 s/659 MB；M 2.63 s/873 MB；
  D 2.45 s/575 MB。容量增加 1.7–4.7 倍参数，没有带来相应的验证收益。
- 最佳 epoch：M 最早（11–13），L/base/D 可到 38；说明不同容量的优化速度不同，
  但最终验证水平接近。

### 6.2 进入第二/三组的结构

选择规则：只看验证集；若多个结构在最佳验证均值的一个标准误（SE）以内，按
**参数最少、计算最便宜**选择。验证均值 SE = std/√3：

| 结构 | 验证均值 | SE | 与最佳的距离 |
| --- | ---: | ---: | ---: |
| D | 0.8385 | 0.0171 | 0 |
| L | 0.8368 | 0.0283 | 0.0017 |
| base | 0.8345 | 0.0248 | 0.0040 |
| M | 0.8203 | 0.0244 | 0.0182 |

D、L、base 都在最佳验证均值的一个 SE 内（最佳 D 的 1 SE 下界为 0.8214）；
M 的 0.8203 略低于该边界，且参数最多、最贵，因此不选。三个候选中选择 **base
（128/4/4）**：它不是验证均值最高的，但差异远小于噪声，且参数最少、训练最快。
事后看 L 的测试均值更高（0.7852），这正说明在 167–208 条验证集上按均值选结构会
受噪声影响；我们没有据此改选 L。

## 7. 预训练编码器微调（第二组）

结构固定为 base（128/4/4），同一划分、同一配方；只改变哪些预训练参数参与训练：

| 实验 | ChemBERTa | ConfSeq | checkpoint 参数 | 可训练参数 | checkpoint 内容 |
| --- | --- | --- | ---: | ---: | --- |
| `base` | 全部冻结（缓存） | 全部冻结（缓存） | 789,121 | 789,121 | 仅融合网络（编码器从缓存/本地加载） |
| `ft_smiles` | 最后 2 层 | 全部冻结（缓存） | 44,893,057 | 14,964,865 | 融合网络 + ChemBERTa；ConfSeq 仍走缓存 |
| `ft_both` | 最后 2 层 | 最后 2 层 | 59,731,073 | 19,696,257 | 融合网络 + 两个编码器 |

如果按“实际参与前向的模型”计，`base` 的总参数是 44,103,936 + 14,838,016 +
789,121 = 59,731,073，其中只有融合网络 789,121 参与训练；预训练编码器不复制进
`base` 的 `model.pt`。`ft_smiles` 的 ConfSeq 仍是缓存的冻结表示，所以它的
checkpoint 不含 ConfSeq 的 14.8 M 参数。

结果：

| 实验 | n | 测试 AUC | 测试 PR-AUC | 测试 F1 | 测试 MCC | 验证 AUC | 训练 AUC | 训练-测试 | 每 epoch 秒 | 峰值显存 MB |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | ---: | ---: |
| `base`（冻结） | 3 | 0.7718 ± 0.0312 | 0.5324 ± 0.0646 | 0.5418 ± 0.0322 | 0.3930 ± 0.0190 | 0.8345 ± 0.0429 | 0.9948 ± 0.0025 | 0.2230 ± 0.0291 | 1.41 | 429 |
| `ft_smiles` | 3 | 0.8265 ± 0.0287 | 0.6548 ± 0.0535 | 0.5920 ± 0.0293 | 0.4549 ± 0.0456 | 0.8461 ± 0.0255 | 0.9946 ± 0.0030 | 0.1681 ± 0.0312 | 3.20 | 991 |
| `ft_both` | 3 | **0.8342 ± 0.0307** | 0.6204 ± 0.0436 | 0.5788 ± 0.0614 | **0.4587 ± 0.0492** | **0.8534 ± 0.0357** | 0.9939 ± 0.0066 | **0.1598 ± 0.0336** | 4.76 | 1475 |

逐 seed：

| 实验 | seed | 验证 AUC | 测试 AUC | PR-AUC | F1 | MCC | 最佳 epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ft_smiles` | 42 | 0.8657 | 0.8057 | 0.6192 | 0.5652 | 0.4028 | 19 |
| `ft_smiles` | 43 | 0.8173 | 0.8145 | 0.6288 | 0.5874 | 0.4874 | 18 |
| `ft_smiles` | 44 | 0.8552 | 0.8593 | 0.7163 | 0.6233 | 0.4745 | 16 |
| `ft_both` | 42 | 0.8651 | 0.8366 | 0.6166 | 0.5556 | 0.4345 | 15 |
| `ft_both` | 43 | 0.8134 | 0.8023 | 0.5788 | 0.5324 | 0.4263 | 28 |
| `ft_both` | 44 | 0.8818 | 0.8636 | 0.6658 | 0.6484 | 0.5153 | 20 |

- 部分微调 ChemBERTa 把测试 AUC 从 0.7718 提到 0.8265（+0.0547），把训练-测试
  差距从 0.223 降到 0.168。这与 V3 `full_ft` 的方向一致（0.7513 → 0.8191），
  但在 V3.1 配方下更高。
- 再解冻 ConfSeq 最后 2 层后为 0.8342（+0.0077，小于一个标准差），验证集和测试集
  都略好，但**没有足够证据说“两个都微调一定优于只微调 ChemBERTa”**。
- 两个微调策略都仍低于 V2 A CatBoost（0.8578 ± 0.0292）：差 0.0313（ft_smiles）
  和 0.0236（ft_both）。ft_both 的验证均值 0.8534 是 V3.1 中最高，但测试均值
  仍低于 A。
- 计算成本：ft_smiles 3.20 s/epoch、991 MB；ft_both 4.76 s/epoch、1475 MB，
  约为冻结 base 的 2.3–3.4 倍时间。
- 梯度检查（seed42）：`smiles_encoder` 梯度在第 1/8/15 epoch 为
  0.150/2.653/4.149，说明解冻层确实在更新；`base` 的缓存路径没有编码器梯度。
  逐层审计（`audit_v3.py` 的 `encoder_grad_detail`）确认：ChemBERTa
  embeddings、pooler、第 0–3 层 `requires_grad=0` 且无梯度；第 4、5 层
  全部参数有梯度，范数分别为 1.129 和 1.206。没有意外更新冻结层。

## 8. 三模态消融（第三组）

结构固定为 base（128/4/4），冻结编码器、缓存 token；只改变输入模态，训练配方相同：

| 实验 | 输入 | n | 测试 AUC | 测试 PR-AUC | 测试 F1 | 测试 MCC | 验证 AUC | 训练 AUC | 训练-测试 |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| `base` | SMILES + ConfSeq + 24 descriptors | 3 | 0.7718 ± 0.0312 | 0.5324 ± 0.0646 | 0.5418 ± 0.0322 | 0.3930 ± 0.0190 | 0.8345 ± 0.0429 | 0.9948 ± 0.0025 | 0.2230 ± 0.0291 |
| `no_smiles` | ConfSeq + 24 descriptors | 3 | **0.8034 ± 0.0342** | **0.5920 ± 0.0488** | **0.5749 ± 0.0506** | **0.4138 ± 0.0729** | 0.8486 ± 0.0338 | 0.9414 ± 0.0564 | **0.1380 ± 0.0863** |
| `no_confseq` | SMILES + 24 descriptors | 3 | 0.7284 ± 0.0063 | 0.4538 ± 0.0054 | 0.5251 ± 0.0244 | 0.3399 ± 0.0234 | 0.8052 ± 0.0684 | 0.9961 ± 0.0043 | 0.2677 ± 0.0039 |
| `no_padel` | SMILES + ConfSeq | 3 | 0.7797 ± 0.0497 | 0.5618 ± 0.1011 | 0.5564 ± 0.0411 | 0.3984 ± 0.0484 | 0.8406 ± 0.0387 | 0.9938 ± 0.0082 | 0.2141 ± 0.0579 |

相对 `base` 的测试 AUC 变化：`no_smiles` **+0.0316**、`no_confseq` **−0.0434**、
`no_padel` **+0.0079**。

- **ConfSeq 是唯一去掉后明显变差的模态**（−0.0434，且训练-测试差升到 0.268）。
  这与 V3 的方向一致；ConfSeq 的 token 流提供了跨 scaffold 的外推信号。
- 去掉 SMILES 仍略好（+0.0316，三套划分中两套提高、一套下降），但差距比 V3
  原始配方的 +0.0553 小；延长训练/改进配方缩小了冻结 SMILES 的负面影响。
- 去掉 24 个 descriptor 的均值略高（+0.0079），但标准差 0.0497，逐 seed 方向
  不一致；在冻结 SMILES 存在时，descriptor 的边际贡献不明确。
- `no_smiles` 的训练 AUC 只有 0.941、训练-测试差 0.138，是四者中过拟合最轻的；
  这再次说明 full 的额外容量/模态主要放大了训练集拟合。
- 三个模态在反向传播中都有非零梯度（第 3.3 节），但“参与训练”不等于“有正边际
  贡献”；消融结果衡量的是后者。

### 8.1 补充：微调后的 SMILES 是否仍有贡献？

`ft_both` 的测试均值 0.8342 明显高于冻结 base 0.7718。为了区分“微调带来的整体
提升”和“SMILES 本身是否仍有贡献”，补跑了 `ft_no_smiles`：ConfSeq 最后 2 层微调、
24 descriptors 保留、完全去掉 SMILES。总参数 15,487,489，可训练 5,380,865。

| 实验 | 输入/微调 | n | 测试 AUC | 测试 PR-AUC | 测试 F1 | 测试 MCC | 验证 AUC | 训练 AUC | 训练-测试 |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| `ft_no_smiles` | ConfSeq 最后 2 层 + descriptors，无 SMILES | 3 | 0.8003 ± 0.0297 | 0.5640 ± 0.0888 | 0.5598 ± 0.0315 | 0.4010 ± 0.0506 | 0.8363 ± 0.0540 | 0.9255 ± 0.0664 | 0.1252 ± 0.0655 |
| `ft_smiles` | ChemBERTa 最后 2 层 + ConfSeq 缓存 + descriptors | 3 | 0.8265 ± 0.0287 | 0.6548 ± 0.0535 | 0.5920 ± 0.0293 | 0.4549 ± 0.0456 | 0.8461 ± 0.0255 | 0.9946 ± 0.0030 | 0.1681 ± 0.0312 |
| `ft_both` | 两个编码器最后 2 层 + descriptors | 3 | **0.8342 ± 0.0307** | 0.6204 ± 0.0436 | 0.5788 ± 0.0614 | 0.4587 ± 0.0492 | 0.8534 ± 0.0357 | 0.9939 ± 0.0066 | 0.1598 ± 0.0336 |
| `no_smiles`（冻结对照） | ConfSeq 缓存 + descriptors | 3 | 0.8034 ± 0.0342 | 0.5920 ± 0.0488 | 0.5749 ± 0.0506 | 0.4138 ± 0.0729 | 0.8486 ± 0.0338 | 0.9414 ± 0.0564 | 0.1380 ± 0.0863 |

- `ft_both` 比 `ft_no_smiles` 高 **+0.0339**，`ft_smiles` 比 `ft_no_smiles` 高
  **+0.0262**：在编码器适配后，SMILES token 流确实提供了增量信息。
- `ft_no_smiles`（0.8003）与冻结 `no_smiles`（0.8034）几乎相同（−0.0031）：
  单独微调 ConfSeq 没有带来提升。增益主要来自 ChemBERTa 的适配，而不是
  “解冻更多参数”本身。
- 这不是严格的单变量消融：`ft_no_smiles` 的 ConfSeq 是微调的，而冻结
  `no_smiles` 的 ConfSeq 是缓存的；但两组的差别只有 0.003，支持上述解释。
- 训练-测试差距：`ft_no_smiles` 0.125 最小，`ft_both` 0.160，`ft_smiles` 0.168；
  微调 SMILES 增加了拟合能力，也带回一部分过拟合，但换来了更高的测试 AUC。

因此，**SMILES 并不是“本质上无用”**：冻结表示下它是负贡献，部分微调后是正贡献。
之前的 V3 结论“去掉 SMILES 更好”只对冻结设置成立。

## 9. SMILES 分支专题

### 9.1 先排除实现错误

- tokenizer 往返：前 5 条 SMILES 解码后与原字符串逐字符一致。
- 截断：最大 298 token < 320，截断 0 条；ConfSeq 最大 499 < 512，截断 0 条。
- 位置嵌入：SMILES 和 ConfSeq 各自有独立的可学习位置 embedding，没有共用位置表。
- padding mask：单元测试验证有效 token 数与 mask 一致、被 mask 的 key 注意力为 0。
- 梯度：SMILES 投影层在 V3.1 base 的一个 batch 上梯度范数 7.54（总梯度 10.53），
  不是“没接上”或“梯度被切断”。
- 注意力：SMILES token 在每层都能看到 ConfSeq/descriptor token，反之亦然。

因此，SMILES 的问题不是工程实现错误，而是**表示质量 + 小数据泛化**。

### 9.2 冻结表示的诊断探针

用均值池化的冻结 token 表示训练逻辑回归（训练集拟合、验证/测试只评估）：

| 表示 | 训练 AUC | 验证 AUC | 测试 AUC |
| --- | ---: | ---: | ---: |
| ChemBERTa SMILES 均值 | 0.9990 | 0.7870 | 0.6631 |
| ConfSeq 均值 | 0.8996 | 0.7807 | 0.7689 |

- 两种表示都能在训练集上被线性分类器几乎完全分开，但 scaffold 测试集上
  ChemBERTa 降到 0.663，ConfSeq 降到 0.769。
- 这说明冻结的 ChemBERTa 表示对训练骨架更“贴合”，跨 scaffold 迁移更差；
  它进入融合网络后仍带着这个问题。
- 两者不是完全重复：测试集线性 CKA 0.475，用 SMILES 均值岭回归预测 ConfSeq
  均值的验证 R² 为 0.416。SMILES 有独立信息，但不足以抵消它的过拟合。

### 9.3 过拟合证据

| 模型 | 训练 AUC | 测试 AUC | 训练-测试 |
| --- | ---: | ---: | ---: |
| V3 `full` seed42 | 0.9963 | 0.6854 | 0.3109 |
| V3.1 `base` seed42 | 0.9922 | 0.7528 | 0.2394 |
| V3.1 `no_smiles` 三 seed 均值 | 0.9414 | 0.8034 | 0.1380 |
| V3.1 `no_confseq` 三 seed 均值 | 0.9961 | 0.7284 | 0.2677 |

去掉 SMILES 同时降低了训练 AUC 和训练-测试差距，并提高测试 AUC；这说明
SMILES 分支是 full 模型过拟合的主要来源之一。注意 `no_confseq` 的训练 AUC
同样很高、测试最低，说明“过拟合”不只属于 SMILES，但 SMILES 是唯一一个
“去掉后测试变好”的模态。

### 9.4 是否由长序列/长分子造成？

不是。

- SMILES 长度均值 28.1、中位数 25、最大 298；绝大多数分子并不长。
- 按测试集 SMILES 长度四分位分组（seed42，base）：

| 长度四分位 | 长度范围 | base AUC | no_smiles AUC | ft_both AUC |
| --- | --- | ---: | ---: | ---: |
| Q1 | 3–18 | 0.720 | 0.707 | 0.776 |
| Q2 | 19–26 | 0.744 | 0.704 | 0.867 |
| Q3 | 27–35 | 0.758 | 0.805 | 0.798 |
| Q4 | 36–168 | 0.740 | 0.822 | 0.887 |

- 长度与误差的 Spearman 相关为负（base −0.116、no_smiles −0.148、ft_both −0.070）：
  长 SMILES 并没有更差，短 SMILES 也不是全部问题。
- 因此“序列太长导致过拟合”不成立；更合理的解释是冻结表示与训练骨架的贴合。

### 9.5 微调后 SMILES 是否仍有贡献？

有。补充实验 `ft_no_smiles`（ConfSeq 最后 2 层微调 + 24 descriptors，去掉 SMILES）
三划分测试 AUC 0.8003 ± 0.0297；保留 SMILES 的 `ft_smiles` 为 0.8265 ± 0.0287，
`ft_both` 为 0.8342 ± 0.0307。也就是说：

- 冻结 SMILES：`base` 0.7718 < `no_smiles` 0.8034（负贡献）；
- 微调 ChemBERTa：`ft_smiles` 0.8265 > `ft_no_smiles` 0.8003（正贡献，+0.0262）；
- 两个编码器都微调：`ft_both` 0.8342 > `ft_no_smiles`（+0.0339）。

结论：**问题在冻结表示，不在 SMILES 模态本身**。V3 的“SMILES 拖累”结论只适用于
冻结 ChemBERTa 的设置；一旦最后 2 层适配到 PPDB 标签，SMILES 的 token 级信息
开始产生正贡献。但这仍不足以超过 CatBoost A（0.8578）。

## 10. 对 12 个问题的明确回答

**1. 当前 V3 的问题是否与模型容量不足有关？**
不是。融合网络从 0.79 M 增到 3.70 M（4.7 倍），验证 AUC 仍在 0.820–0.839
之间，测试 AUC 0.764–0.785，没有随容量单调提高；M（最大）的验证均值反而最低。
训练 AUC 已经 0.963–0.995，说明模型有能力拟合训练集，问题是泛化而不是容量。

**2. 增加 Transformer 层数是否改善了表现？**
没有稳定改善。6 层 L 的验证均值 0.8368（base 0.8345），8 层 D 为 0.8385，
差异都小于 seed 标准差（0.03–0.05）。D 的测试均值 0.7639 反而低于 base。
L 的测试均值 0.7852 最高，但这是事后观察，验证集不支持“6 层更好”。

**3. 增加隐藏维度是否改善了表现？**
没有。256 维 M 的验证均值 0.8203、测试 0.7813，均不高于 128 维 base
（0.8345/0.7718）；M 的参数量是 base 的 4.7 倍、显存 2 倍。

**4. 增加训练轮数是否改善了表现？**
有一点，但不是主因。V3.1 base（最多 100 epoch、patience 18）三划分测试均值
0.7718，比 V3 `full`（20 epoch、patience 5）的 0.7513 高 0.0205；但配方同时
改了 AMP、warmup/cosine 和有效 batch，不能把差异单独归给轮数。最佳 epoch 为
11–38，说明 20 epoch 对部分 run 确实偏短；但验证损失在后期普遍升到 1.5–3.0，
更多轮数主要是在拟合训练骨架。**延长训练不是解决泛化的办法。**

**5. GPU 混合精度是否正常工作？**
是。`torch==2.14.1+cu130`、RTX 4060 Laptop、`torch.cuda.is_available()=True`；
ChemBERTa/ConfSeq/融合网络全部在 `cuda:0`；BF16 autocast 前向+反向通过，
损失在 FP32 计算；所有模块梯度非零。冻结模型 1.4 s/epoch，`ft_both`
4.8 s/epoch，峰值显存 0.43–1.48 GB，没有 OOM、没有退回 CPU。

**6. SMILES 分支为什么可能拖累结果？**
不是 tokenizer、截断、位置编码或 mask 的 bug（第 9.1 节）。原因是冻结的
ChemBERTa 表示在 scaffold 外推上较弱：均值池化线性探针训练 AUC 0.999、
测试只有 0.663（ConfSeq 为 0.769）；进入融合后，V3 full 的训练 AUC 0.988–0.996、
测试 0.685–0.788，V3.1 去掉 SMILES 后训练 AUC 均值降到 0.941、测试升到 0.8034。
也不是长序列问题：SMILES 平均 28 token、最大 298、截断 0，长度与误差负相关。
部分微调 ChemBERTa 后，`ft_smiles` 0.8265、`ft_both` 0.8342，高于去掉 SMILES 的
`ft_no_smiles` 0.8003——**冻结表示才是拖累的来源，SMILES 模态本身不是。**

**7. ConfSeq 是否持续提供有效帮助？**
是。冻结消融中去掉 ConfSeq 降到 0.7284（−0.0434），是唯一明显变差的模态；
ConfSeq 均值探针测试 AUC 0.769，高于 SMILES 的 0.663。`ft_no_smiles`
（ConfSeq 微调 + descriptors）仍有 0.8003，说明 ConfSeq 是较可靠的主干。
但“微调 ConfSeq”本身没有额外收益（0.8003 vs 冻结 no_smiles 0.8034）。

**8. PaDEL 是否提供独立的信息？**
证据较弱。冻结消融中去掉 24 descriptors 后测试 AUC 从 0.7718 升到 0.7797
（+0.0079，标准差 0.0497），方向不一致；descriptor 编码器的梯度在多数 run 中
比 SMILES/ConfSeq 投影小 1–2 个数量级，注意力份额也只有 5%–7%。
但在 `no_smiles`（ConfSeq + descriptors）配置中，24 个 token 与 ConfSeq 一起
达到 0.8034，说明它至少是条件性有用的；不能据此断言“完全独立”。

**9. 三种信息是否都真正参与了反向传播？**
是。`full` 的梯度探针：SMILES 投影 0.50、ConfSeq 投影 0.40、descriptor 编码器
0.026（V3 seed42）；V3.1 base 为 7.54/3.79/0.55。三者都非零。冻结编码器没有
梯度；`ft_smiles` 只解冻最后 2 层，逐层审计显示 layer 0–3/embeddings/pooler
无梯度，layer 4/5 梯度范数 1.129/1.206。没有 detach 或错误缓存切断计算图。

**10. 共同注意力是否存在跨模态信息交流？**
存在。每层每头的注意力都保存；以 V3.1 base 最后一层为例（份额，其余主要给
CLS）：SMILES→ConfSeq 0.118、ConfSeq→SMILES 0.061、descriptors→SMILES 0.061、
descriptors→ConfSeq 0.115；L 的 SMILES→ConfSeq 为 0.139。
但约 75% 的注意力流向 CLS，模态间份额只有 5%–16%；注意力只能说明计算时的
信息参考，不能证明模型定位到真实化学作用位点。

**11. 模型是否存在明显过拟合？**
是。V3 `full` seed42 训练 AUC 0.9963、测试 0.6854（差 0.311）；V3.1 base
三 seed 平均训练 AUC 0.9948、测试 0.7718（差 0.223）；`no_smiles` 差 0.138。
验证损失普遍在最佳 AUC 之后继续上升（常到 1.5–3.0），说明早停按 AUC 选出的
checkpoint 校准很差；阈值只能在验证集补偿。三套 scaffold 划分上方向一致。

**12. 哪个结构在性能、稳定性和计算成本之间最合理？**
- 冻结设置：**base（128/4/4）**。验证均值与 L/D 相差 <0.005，参数 0.79 M、
  1.41 s/epoch、429 MB，最便宜。
- 追求测试性能：**ft_both**（0.8342）或 **ft_smiles**（0.8265），但训练时间
  2.3–3.4 倍、checkpoint 44–60 M 参数；两者差异 0.008，小于 seed 标准差。
- 计算最便宜且效果可接受：`ft_no_smiles` 0.8003，但放弃了 SMILES 模态，
  不符合“三模态统一”的目标。
- CatBoost A 仍是整体最佳（0.8578）。

## 11. 结论与建议

1. **继续使用 128 维、4 层、4 头**。没有证据支持 192 或 256 维，也没有证据
   支持 8 层；这些配置增加了 1.7–4.7 倍参数和 1.5–2 倍显存，验证/测试没有
   可复现的收益。
2. **不建议继续增加 Transformer 深度**。当前融合层梯度随深度快速衰减
   （V3 seed42：第 0 层 0.429 → 第 3 层 0.053），加深主要增加优化难度。
3. **微调策略需要改变**：如果允许 2–3 倍训练成本，解冻 ChemBERTa 最后 2 层
   （可再加 ConfSeq）是唯一稳定的增益（+0.055–0.062 AUC）；如果必须冻结，
   128/4/4 + 缓存是最合理的选择。
4. **SMILES 不应被删除**。冻结时它是负贡献，微调后是正贡献；研究重点应是
   表示适配，而不是把三模态退回两模态。
5. **主要瓶颈不是容量，而是数据与泛化**：431 个阳性、167–208 条验证集、
   scaffold 外推、阈值不稳定、预训练语料重叠未知。继续堆参数不会解决这些问题。
6. **V3.1 仍没有超过 CatBoost A**。统一多模态模型的最好结果 0.8342 低于 A 的
   0.8578；A 仍是最可靠的基线。V3.1 的贡献是 GPU 训练、可审计的学习行为、
   以及把“SMILES 拖累”定位到冻结表示而不是模态本身。
7. **外部验证同样支持保留 A**：21 个未参与训练的已知分子上，CatBoost AUC
   0.750，V3.1 最好的 `ft_no_smiles` 0.673，`ft_both` 0.558；在 10 个
   scaffold-disjoint 分子上所有模型都接近/低于随机。内部测试提升没有转化为
   外部泛化，不能用于实验筛选。

### 11.1 解释边界

- 三套 scaffold 划分的测试集彼此有重叠，mean ± std 不是三次独立复现；
  它们只表示对不同 scaffold 切分的敏感性。
- 标签是 PPDB 的杀菌剂分类，不是 EC50/MIC/实验抑制率；阴性也不等于实验无活性。
- V3.1 已补做 21 个外部已知分子的初步验证（第 11.2 节），但样本只有 8 个阳性，
  且 11/21 与训练集共享 Murcko scaffold；结果不支持 V3.1 的外部泛化。
- 结构选择按验证集 one-SE + 最简规则；事后 L 的测试均值更高，但我们没有据此
  回选，以避免测试集泄漏到模型选择。
- 早停按验证 ROC-AUC，验证损失在后期常升到 1.5–3.0；选出的 checkpoint 校准
  较差，阈值在 167–208 条验证集上不稳定。
- 只解冻最后 2 层，没有做全量微调或逐层学习率搜索；`ft_no_smiles` 是补充实验，
  不是严格单变量消融。
- 预训练语料重叠状态仍是 `unknown`；不能声称 ChemBERTa/ConfSeq 预训练集
  与 PPDB 分子没有重叠。
- 注意力份额只是计算时的信息参考，不能解释为化学作用位点。

### 11.2 外部已知分子验证（补充）

`models/external_validation.csv` 的 21 个分子（8 个已知杀菌剂、13 个对照）没有参与
训练；InChIKey 与训练集重叠为 0。为了用 V3.1 预测，重新为这些分子生成了：

- 官方 ConfSeq 字符串：`scripts/confseq_pipeline.py`，21/21 成功，token 全部在
  词表内（unknown 0/1091）；
- 24 个 PaDEL 描述符：`padelpy` 2D 计算，21/21 完成，无 NaN。

注意：这不是干净的 scaffold 外部集——21 个分子中有 **11 个的 Murcko scaffold
出现在训练集**。因此全部 21 个的指标偏乐观；下面同时给出 10 个 scaffold-disjoint
分子的敏感性分析。

**全部 21 个（8 阳性 / 13 对照，阈值 0.5）：**

| 模型 | AUC | PR-AUC | 杀菌剂 ≥0.5 | 对照 ≥0.5 | 杀菌剂中位数 | 对照中位数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CatBoost combined（旧基线） | **0.750** | **0.745** | 5/8 | 2/13 | 0.560 | 0.328 |
| V3.1 `base` | 0.596 | 0.451 | 4/8 | 4/13 | 0.464 | 0.337 |
| V3.1 `no_smiles` | 0.644 | 0.581 | 4/8 | 4/13 | 0.484 | 0.307 |
| V3.1 `ft_smiles` | 0.462 | 0.398 | 3/8 | 8/13 | 0.347 | 0.549 |
| V3.1 `ft_both` | 0.558 | 0.495 | 4/8 | 7/13 | 0.469 | 0.634 |
| V3.1 `ft_no_smiles` | 0.673 | 0.605 | 5/8 | 7/13 | 0.673 | 0.505 |

**10 个 scaffold-disjoint 分子（5 阳性 / 5 对照；敏感性分析，n 很小）：**

| 模型 | AUC | PR-AUC | 杀菌剂 ≥0.5 | 对照 ≥0.5 |
| --- | ---: | ---: | ---: | ---: |
| CatBoost combined | 0.520 | 0.672 | 3/5 | 1/5 |
| V3.1 `base` | 0.120 | 0.381 | 2/5 | 4/5 |
| V3.1 `no_smiles` | 0.440 | 0.609 | 3/5 | 2/5 |
| V3.1 `ft_smiles` | 0.040 | 0.361 | 1/5 | 5/5 |
| V3.1 `ft_both` | 0.080 | 0.371 | 2/5 | 5/5 |
| V3.1 `ft_no_smiles` | 0.280 | 0.544 | 2/5 | 5/5 |

逐分子结果在 `experiments/v3_1/external/external_predictions.csv`，汇总在
`external_summary.json`。8 个已知杀菌剂的概率（V3.1 为三 seed 均值）：

| 杀菌剂 | CatBoost | base | ft_both | ft_no_smiles |
| --- | ---: | ---: | ---: | ---: |
| phthalide | 0.870 | 0.980 | 0.991 | 0.907 |
| pyroxyfur | 0.721 | 0.045 | 0.432 | 0.433 |
| fluoxapiprolin | 0.676 | 0.351 | 0.678 | 0.711 |
| ipflufenoquin | 0.594 | 0.335 | 0.369 | 0.341 |
| pyrimorph | 0.526 | 0.577 | 0.505 | 0.635 |
| diclocymet | 0.365 | 0.664 | 0.364 | 0.830 |
| metyltetraprole | 0.237 | 0.923 | 0.945 | 0.927 |
| isoprothiolane | 0.085 | 0.009 | 0.017 | 0.151 |

观察：

- 全部 21 个上，CatBoost 仍是最好（AUC 0.750）；V3.1 最好的外部配置是
  `ft_no_smiles` 0.673，`base`/`no_smiles` 约 0.60–0.64，`ft_smiles` 0.462、
  `ft_both` 0.558。内部测试 AUC 高的模型没有带来外部优势。
- `ft_smiles`/`ft_both` 在外部集上把 8/13 或 7/13 个对照排到 ≥0.5，
  特别是 warfarin、tolfenpyrad、omeprazole 等；这是明显的假阳性。
- 在 10 个 scaffold-disjoint 分子上，CatBoost 也只有 0.520；V3.1 全部 ≤0.44，
  多数模型把对照排得比杀菌剂高。**没有证据表明这些模型能对真正的新骨架
  做可靠外推。**
- 8 个杀菌剂中，phthalide、metyltetraprole 在 V3.1 上分数较高；
  isoprothiolane、pyroxyfur 仍被漏掉；不同模型的错误方向不一致。
- 样本只有 8 个阳性、5 个 scaffold-disjoint 阳性，AUC 的置信区间很宽；
  不能据此做显著性结论，只能作为“外部泛化失败”的警告。

结论：V3.1 的内部 scaffold 测试提升没有转化为外部已知分子的排序能力；
CatBoost A 仍是外部验证最好的模型。部署或实验筛选仍不应使用 V3.1 分数。

## 12. 复现命令

```bash
cd D:/works/yuce

# 1) GPU 核验（应看到 cuda:0、bf16、全部模块在 GPU、反向梯度非零）
./.venv/Scripts/python.exe scripts/check_v3_1_gpu.py

# 2) 结构实验：base / L / M / D，seed 42/43/44
./.venv/Scripts/python.exe scripts/run_v3_1_experiments.py \
  --experiments base,L,M,D --seeds 42,43,44 --epochs 100 --patience 18 \
  --batch-size 16 --grad-accum 2 --lr 3e-4 --dropout 0.3 --amp bf16 \
  --out experiments/v3_1/group1

# 3) 微调：ChemBERTa 最后 2 层 / 两个编码器最后 2 层
./.venv/Scripts/python.exe scripts/run_v3_1_experiments.py \
  --experiments ft_smiles,ft_both --arch base --seeds 42,43,44 \
  --epochs 100 --patience 18 --batch-size 16 --grad-accum 2 \
  --lr 3e-4 --encoder-lr 2e-5 --dropout 0.3 --amp bf16 \
  --out experiments/v3_1/group2

# 4) 消融：no_smiles / no_confseq / no_padel
./.venv/Scripts/python.exe scripts/run_v3_1_experiments.py \
  --experiments no_smiles,no_confseq,no_padel --arch base --seeds 42,43,44 \
  --epochs 100 --patience 18 --batch-size 16 --grad-accum 2 \
  --lr 3e-4 --dropout 0.3 --amp bf16 --out experiments/v3_1/group3

# 5) 补充：微调 ConfSeq、去掉 SMILES
./.venv/Scripts/python.exe scripts/run_v3_1_experiments.py \
  --experiments ft_no_smiles --arch base --seeds 42,43,44 \
  --epochs 100 --patience 18 --batch-size 16 --grad-accum 2 \
  --lr 3e-4 --encoder-lr 2e-5 --dropout 0.3 --amp bf16 \
  --out experiments/v3_1/smiles_check

# 6) 审计 V3 与微调模型
./.venv/Scripts/python.exe scripts/audit_v3.py \
  --model experiments/v3/runs/V3_full_seed42 \
  --out experiments/v3_1/audit --device cpu --attention-samples 64
./.venv/Scripts/python.exe scripts/audit_v3.py \
  --model experiments/v3_1/group2/runs/V3_1_ft_smiles_seed42 \
  --out experiments/v3_1/audit --device cpu --attention-samples 16 --skip-probes

# 7) SMILES 长度/误差分析
./.venv/Scripts/python.exe scripts/analyze_smiles_effect.py \
  --runs experiments/v3_1/group1/runs/V3_1_base_seed42 \
         experiments/v3_1/group3/runs/V3_1_no_smiles_seed42 \
         experiments/v3_1/group2/runs/V3_1_ft_smiles_seed42 \
         experiments/v3_1/group2/runs/V3_1_ft_both_seed42 \
  --out experiments/v3_1/smiles_analysis

# 8) 外部 21 个已知分子（需先生成 ConfSeq/PaDEL，见第 11.2 节）
./.venv/Scripts/python.exe scripts/external_validate_v3_1.py \
  --input experiments/v3_1/external/external_features.csv \
  --out experiments/v3_1/external --device cuda

# 9) 预测
molpredict v3-predict --model experiments/v3_1/group2/runs/V3_1_ft_both_seed44 \
  --input new_molecules.csv --out predictions.csv

# 10) 测试
./.venv/Scripts/python.exe -m pytest -q
```

