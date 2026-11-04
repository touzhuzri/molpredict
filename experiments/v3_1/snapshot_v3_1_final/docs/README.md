# molpredict — 轻量级分子性质预测工具

基于分子结构预测分子性质。第一阶段的目标是判断分子**是否可能具有杀菌活性**。

> 说明：训练出的模型输出是统计意义上的预测分数，不是真实实验活性概率。
> 对没有真实标签的分子不会伪造标签；没有标签数据时不会训练模型。

核心流程：

```
分子结构输入 → 分子结构处理 → 特征提取 → 模型训练 → 分子性质预测
```

## 当前进度

- [x] **M0 项目初始化**：目录、依赖、命令行入口
- [x] **M1 分子数据处理**：CSV 批量输入，生成 InChIKey、2D PNG、3D SDF，输出成功/失败记录
- [x] **M2 特征处理**：Morgan 指纹、RDKit 描述符、ConfSeq 256 维 embedding
- [x] **M3 模型训练**：CatBoost 二分类，四种特征组合对比
- [x] **M4 性质预测**：输入 SMILES，输出预测分数
- [x] **V2 多模态实验**：ChemBERTa + ConfSeq + PaDEL/RDKit，A–G 对比与三套 scaffold 稳定性（实验性，未替换基线）
- [x] **V3 统一多模态 Transformer**：token 级共同注意力 + 三组消融 + 预测接口（实验性，未替换基线）

## 目录结构

```
yuce/
├── pyproject.toml          # 项目元数据、依赖、命令行入口
├── README.md
├── .gitignore
├── src/molpredict/
│   ├── __init__.py         # 包版本
│   ├── __main__.py         # 支持 python -m molpredict
│   ├── cli.py              # 命令行入口（process/train/predict/--check）
│   ├── config.py           # 项目路径配置
│   ├── molecules.py        # CSV → InChIKey / 2D PNG / 3D SDF
│   ├── modeling.py         # 特征、CatBoost 训练、预测
│   ├── padel.py            # APPf Table S2 的 24 个 PaDEL 描述符
│   ├── v2.py               # V2 多模态网络、缓存、训练与评估
│   ├── v3_encoders.py      # ChemBERTa/ConfSeq token 编码与缓存
│   ├── v3_model.py         # V3 统一 token 序列与共享 Transformer
│   ├── v3_data.py          # V3 缓存校验、collate、标准化、泄漏审计
│   ├── v3_train.py         # V3 训练、早停、checkpoint 与指标
│   ├── v3_predict.py       # V3 预测接口
│   └── v3_1_train.py       # V3.1 GPU/AMP、warmup-cosine、诊断与注意力审计
├── scripts/
│   ├── fetch_ppdb.py       # 抓取 PPDB 分类标签与结构
│   ├── confseq_pipeline.py # 3D → ConfSeq 字符串（调用官方 demo 代码）
│   ├── run_v2_experiments.py # V2 缓存、划分、A–G 实验与稳定性汇总
│   ├── run_v3_experiments.py # V3 主实验、三组消融、部分解冻
│   ├── run_v3_1_experiments.py # V3.1 结构/微调/消融实验
│   ├── check_v3_1_gpu.py     # CUDA/AMP/模块设备实测
│   ├── audit_v3.py           # 学习行为审计（梯度/注意力/探针）
│   ├── analyze_smiles_effect.py # SMILES 长度/误差诊断
│   ├── summary_v3_1_tables.py   # V3.1 结果表生成
│   └── aggregate_v3_runs.py  # 合并并行 V3 输出
├── docs/
│   ├── v2_architecture.md  # V2 架构、预训练权重来源、运行命令
│   ├── v3_architecture.md  # V3 统一 token 架构、冻结策略、许可证
│   ├── v3_report.md        # V3 实验报告、消融与 V2 对比
│   └── v3_1_report.md      # V3.1 GPU、加深网络、审计与 12 问回答
├── data/
│   ├── raw/                # 原始输入数据
│   ├── processed/          # 处理后分子（2D/3D）
│   ├── training/           # 带标签数据、ConfSeq 与 embedding
│   └── external/           # ConfSeq 仓库、PPDB 页面缓存（不提交）
├── experiments/v2/         # 缓存、scaffold 划分、各组运行产物与汇总
├── experiments/v3/         # V3 缓存、划分、15 个正式 run 与汇总
├── experiments/v3_1/       # V3.1 结构/微调/消融、审计与 GPU 核验
├── models/                 # 训练后的 .cbm 模型、指标、测试集预测
└── tests/
    ├── test_smoke.py       # 最小冒烟测试
    ├── test_v3.py          # V3 结构、mask、冻结、保存/预测测试
    └── test_v3_1.py        # V3.1 embed、注意力、梯度组、调度器测试
```

## 安装

需要 Python 3.10+（已在 Python 3.10.11 上验证）。

```bash
cd D:/works/yuce

# 创建虚拟环境（Windows Git Bash）
/c/Users/Admin/AppData/Local/Programs/Python/Python310/python.exe -m venv .venv
source .venv/Scripts/activate

# 安装项目及核心依赖（RDKit、scikit-learn、numpy、CatBoost）
pip install -e .

# 若下载缓慢，可临时指定代理（按实际端口修改）
# pip install --proxy http://127.0.0.1:7897 -e .

# 开发依赖（pytest）
pip install -e ".[dev]"

# 可选：ConfSeq embedding（torch、transformers 等）
pip install -e ".[confseq]"
```

## 运行

```bash
# 打印项目信息（首次运行会创建 data/、models/ 目录）
molpredict

# 或
python -m molpredict

# 检查 Python 与核心依赖
molpredict --check

# 批量处理分子（CSV 至少含 name、smiles 两列）
molpredict process data/raw/molecules.csv

# 指定输出目录（可选，默认 data/processed）
molpredict process data/raw/molecules.csv --out data/processed
```

输出结构：

```
data/processed/
├── molecules.csv          # 成功记录：record_id, name, smiles, inchikey, image_2d
├── failed.csv             # 失败记录：name, smiles, reason
├── structures_2d/
│   └── molecule_001.png ...
└── structures_3d/
    └── molecules.sdf      # 多条 3D 构象，含名称与 InChIKey 属性
```

处理失败或重复 InChIKey 的分子写入 `failed.csv`，不中断其余分子。

## 训练

训练数据来自 PPDB，来源、标签定义和数量见 [`data/training/DATA_SOURCES.md`](data/training/DATA_SOURCES.md)。

```bash
# 二维特征（Morgan / 描述符 / 组合）
molpredict train data/training/ppdb_labeled.csv --out models --features morgan,descriptors,combined

# ConfSeq embedding（需先生成 embedding，见下）
molpredict train data/training/confseq.csv --out models \
  --features confseq,all \
  --embeddings data/training/confseq_embeddings.npy
```

划分方式：Murcko scaffold 分组（`GroupShuffleSplit`，80/20），避免同骨架分子同时进入训练集和测试集。
训练集内部再划 15% 做早停，测试集只用于最终评估。

## ConfSeq 接入

使用官方仓库 `https://github.com/jiachengxiong/ConfSeq`：

- `scripts/confseq_pipeline.py` 调用官方 `demo/ConfSeq.py`，把 RDKit 3D 构象转成 ConfSeq 字符串；
- 官方 `representation_learning/get_confseq_embeddings.py` 生成 256 维 embedding。

```bash
python scripts/confseq_pipeline.py --csv data/training/ppdb_labeled.csv \
  --out data/training/confseq.csv --workers 8

python data/external/ConfSeq/representation_learning/get_confseq_embeddings.py \
  --input_csv data/training/confseq.csv --confseq_column confseq \
  --checkpoint data/external/ConfSeq/representation_learning/checkpoints/model_epoch_1.pth \
  --output_npy data/training/confseq_embeddings.npy --batch_size 128 --device cpu
```

说明：官方 OneDrive 共享链接匿名访问返回 401，实际 checkpoint 来自 Hugging Face 镜像
`Oopstom/confseq-shape-screen/model_epoch_1.pth`，可用官方代码 `strict=True` 完整加载。
ConfSeq 仓库和权重放在 `data/external/`，不纳入版本管理。转换成功 1823/1827。

## 预测

```bash
molpredict predict \
  --smiles "CC(C)(C)C(=O)C(n1cncn1)Oc1ccc(Cl)cc1" \
  --model models/combined_catboost.cbm \
  --feature combined
```

输出预测分数与阈值判断。该分数是模型输出，不是实验活性概率。

## 测试集结果

同一 scaffold 划分、1823 个分子（431 阳性 / 1392 阴性，测试集 323 / 80 阳性）：

| 特征 | ROC-AUC | PR-AUC | F1 | Precision | Recall |
| --- | --- | --- | --- | --- | --- |
| Morgan (2048) | 0.8485 | 0.6534 | 0.6374 | 0.5686 | 0.7250 |
| RDKit 描述符 | 0.8621 | 0.6863 | 0.6065 | 0.6267 | 0.5875 |
| Morgan + 描述符 | **0.8778** | **0.7307** | **0.6918** | **0.6962** | 0.6875 |
| ConfSeq embedding | 0.7998 | 0.5353 | 0.5350 | 0.5455 | 0.5250 |
| Morgan + 描述符 + ConfSeq | 0.8409 | 0.6273 | 0.5584 | 0.5811 | 0.5375 |

完整 1827 分子上的二维模型指标另存于 `models/metrics_full1827.json`。
测试集逐条预测见 `models/predictions_*.csv`。

## V2 多模态模型

V2 在保留上述 CatBoost 基线的前提下，增加统一多模态网络：ChemBERTa SMILES 编码器、
官方 ConfSeq 编码器和数值描述符分支分别投影到 128 维，经注意力池化后用门控或
Transformer 注意力融合，再经分类头输出杀菌剂分类分数。预训练编码器默认冻结；另做
ChemBERTa 最后 2 层部分解冻对照。完整架构、权重来源和限制见
[`docs/v2_architecture.md`](docs/v2_architecture.md)。

```bash
# 安装 V2 依赖（padelpy 需要 Java；已有环境可跳过）
pip install -e ".[confseq,v2]"

# 生成/校验缓存：ChemBERTa token states、ConfSeq token states、PaDEL 24、RDKit 217
./.venv/Scripts/python.exe scripts/run_v2_experiments.py --prepare

# A–G，三套 Murcko scaffold 划分，E/F/G 同时比较 gated / attention 融合
./.venv/Scripts/python.exe scripts/run_v2_experiments.py \
  --experiments A,B,C,D,E,F,G --seeds 42,43,44 \
  --fusions attention,gated --epochs 60 --patience 12

# F 的 ChemBERTa 最后 2 层部分解冻
./.venv/Scripts/python.exe scripts/run_v2_experiments.py \
  --experiments F --seeds 42 --fusions attention --finetune
```

| 实验 | 输入 |
| --- | --- |
| A | Morgan 2048 + RDKit 217，CatBoost 基线 |
| B | ChemBERTa token states |
| C | 官方 ConfSeq token states |
| D | PaDEL 24（APPf Table S2） |
| E | ChemBERTa + ConfSeq |
| F | ChemBERTa + ConfSeq + PaDEL 24 |
| G | ChemBERTa + ConfSeq + RDKit 217 |

三套 scaffold 划分（seed 42/43/44）的测试均值 ± 标准差：

| 配置 | ROC-AUC | PR-AUC | F1 | MCC |
| --- | --- | --- | --- | --- |
| A：Morgan + RDKit 217，CatBoost | **0.8578 ± 0.0292** | **0.6985 ± 0.0285** | **0.6111 ± 0.0513** | **0.4741 ± 0.0566** |
| B：ChemBERTa | 0.7579 ± 0.0553 | 0.5299 ± 0.0907 | 0.5344 ± 0.0336 | 0.3586 ± 0.0447 |
| C：ConfSeq | 0.7803 ± 0.0193 | 0.5835 ± 0.0294 | 0.5398 ± 0.0427 | 0.3723 ± 0.0595 |
| D：PaDEL 24 | 0.6343 ± 0.0260 | 0.3547 ± 0.0021 | 0.4267 ± 0.0256 | 0.1823 ± 0.0390 |
| E：ChemBERTa + ConfSeq，gated | 0.8044 ± 0.0309 | 0.6251 ± 0.0742 | 0.5563 ± 0.0199 | 0.3896 ± 0.0319 |
| F：ChemBERTa + ConfSeq + PaDEL 24，gated | 0.7864 ± 0.0231 | 0.5309 ± 0.0370 | 0.5627 ± 0.0270 | 0.3965 ± 0.0612 |
| G：ChemBERTa + ConfSeq + RDKit 217，attention | 0.7876 ± 0.0397 | 0.5785 ± 0.0635 | 0.5659 ± 0.0482 | 0.4331 ± 0.0592 |
| F-ft：F + ChemBERTa 最后 2 层解冻，attention | 0.8159 ± 0.0197 | 0.6053 ± 0.0078 | 0.5805 ± 0.0455 | 0.4237 ± 0.0665 |

E/F/G 的 attention 版本、每套划分明细、混淆矩阵和部分解冻结果见
[`docs/v2_report.md`](docs/v2_report.md)。当前 **CatBoost A 仍是最好的配置**；部分
解冻的 F-ft 明显优于冻结版 F，但在 seed42 上仍未超过同一划分的 A，暂不能替换基线。
这些数字是 PPDB 分类标签上的模型指标，不是实验杀菌活性。

## V3 统一多模态 Transformer

V3 在 token 级把 SMILES、ConfSeq 和 24 个 PaDEL 描述符 token 放进**同一个 Transformer**
（不是三个模型投票，也不是先池化再拼三个向量）：

```text
[CLS] + ChemBERTa SMILES tokens + ConfSeq tokens + 24 descriptor tokens
      → shared 4-layer Transformer → CLS → 一个分类分数
```

- SMILES：ChemBERTa tokenizer/encoder，最大长度 320（V3 实测最大 298，无截断）。
- ConfSeq：官方编码器/权重，token 级表示，最大长度 512。
- PaDEL：24 个独立 token，每个由“标准化数值 + 特征身份 embedding”组成。
- 默认冻结两个预训练编码器；`full_ft` 额外解冻 ChemBERTa 最后 2 层。
- 数据/划分完全复用 V2 的 scaffold seed 42/43/44。

```bash
# 安装（复用 V2 的 confseq/v2 extras；需要 Java 以运行 PaDEL）
pip install -e ".[confseq,v2]"

# 生成 V3 320-token SMILES 缓存，并校验 V2 ConfSeq/PaDEL 缓存
./.venv/Scripts/python.exe scripts/run_v3_experiments.py --prepare

# 主实验 + 三组消融，三套 scaffold 划分
./.venv/Scripts/python.exe scripts/run_v3_experiments.py \
  --experiments full,no_smiles,no_confseq,no_padel \
  --seeds 42,43,44 --epochs 20 --patience 5 --lr 3e-4 --dropout 0.3

# 可选：解冻 ChemBERTa 最后 2 层（编码器 lr=2e-5）
./.venv/Scripts/python.exe scripts/run_v3_experiments.py \
  --experiments full_ft --seeds 42,43,44 --epochs 12 --patience 4 \
  --batch-size 32 --lr 3e-4 --encoder-lr 2e-5 --dropout 0.3
```

结果在 `experiments/v3/summary.csv` 和 `experiments/v3/stability.json`；架构见
`docs/v3_architecture.md`，完整实验报告见 `docs/v3_report.md`。

三套 scaffold 划分的测试均值 ± 标准差（`full_ft` 使用 batch=32、12 epochs，
与冻结实验配方不同）：

| 配置 | ROC-AUC | PR-AUC | F1 | MCC |
| --- | --- | --- | --- | --- |
| `full`：SMILES + ConfSeq + 24 PaDEL | 0.7513 ± 0.0571 | 0.5014 ± 0.0589 | 0.4689 ± 0.0652 | 0.2981 ± 0.0544 |
| `no_smiles`：ConfSeq + 24 PaDEL | **0.8066 ± 0.0222** | **0.6024 ± 0.0695** | **0.5682 ± 0.0428** | **0.4033 ± 0.0576** |
| `no_confseq`：SMILES + 24 PaDEL | 0.7155 ± 0.0749 | 0.4612 ± 0.0474 | 0.4741 ± 0.0723 | 0.2530 ± 0.1034 |
| `no_padel`：SMILES + ConfSeq | 0.7850 ± 0.0248 | 0.5719 ± 0.0402 | 0.5495 ± 0.0126 | 0.3830 ± 0.0294 |
| `full_ft`（ChemBERTa 最后 2 层） | 0.8191 ± 0.0210 | 0.6223 ± 0.0296 | 0.5825 ± 0.0312 | 0.4240 ± 0.0399 |

V3 的统一融合没有超过 V2：冻结 `full` 低于 V2 F（0.7602），最好的冻结三划分 V3
配置 `no_smiles`（0.8066）低于 V2 A CatBoost（0.8578）；`full_ft`（0.8191）与
V2 F-ft（0.8159）基本持平，仍低于 A。**CatBoost A 仍是当前最可靠的基线，
V3 不替换它。**

## V3.1：GPU、加深网络与学习行为审计

V3.1 保留同一套统一 token 序列结构，改为 GPU BF16、warmup+cosine、
batch 16×2 梯度累积、最多 100 epoch/patience 18，并测试更深/更宽的结构和
部分微调。V3 原始结果与代码快照保留在 `experiments/v3/`，V3.1 在
`experiments/v3_1/`，完整报告见 `docs/v3_1_report.md`。

三套 scaffold 划分的测试 AUC（均值 ± 标准差，n=3）：

| 配置 | 测试 AUC | 说明 |
| --- | --- | --- |
| `base`（128/4/4，冻结） | 0.7718 ± 0.0312 | V3 结构在新配方下的基线 |
| `L`（192/6/4，冻结） | 0.7852 ± 0.0115 | 验证集无显著提升，测试较高但不能据此回选 |
| `M`（256/6/8，冻结） | 0.7813 ± 0.0097 | 参数 4.7 倍，验证均值反而最低 |
| `D`（128/8/4，冻结） | 0.7639 ± 0.0361 | 加深无稳定收益 |
| `no_smiles`（冻结） | 0.8034 ± 0.0342 | 去掉冻结 SMILES 仍略好 |
| `no_confseq`（冻结） | 0.7284 ± 0.0063 | 去掉 ConfSeq 明显变差 |
| `no_padel`（冻结） | 0.7797 ± 0.0497 | 24 描述符边际贡献不明确 |
| `ft_smiles` | 0.8265 ± 0.0287 | ChemBERTa 最后 2 层 |
| `ft_both` | **0.8342 ± 0.0307** | 两个编码器最后 2 层 |
| `ft_no_smiles` | 0.8003 ± 0.0297 | 补充：微调 ConfSeq、去掉 SMILES |

- 增加深度/宽度没有可复现的验证收益；按验证集 one-SE + 最简规则选 128/4/4。
- 部分微调 ChemBERTa 是唯一稳定的增益（+0.055），两个都微调为 +0.062；
  仍低于 CatBoost A（0.8578）。
- V3 的“SMILES 拖累”在 V3.1 中被定位为**冻结表示**问题：冻结时去掉 SMILES
  更好，微调后保留 SMILES 更好（`ft_both` 0.8342 vs `ft_no_smiles` 0.8003）。
- 学习行为审计：训练 AUC 0.94–0.99，训练-测试差 0.14–0.31；三种模态都有非零
  梯度；跨模态注意力存在但约 75% 流向 CLS；descriptor 分支梯度明显偏小。
  GPU 核验与逐轮曲线见 `experiments/v3_1/gpu_check.json`、`audit/`。

V3.1 命令：

```bash
# GPU 核验
./.venv/Scripts/python.exe scripts/check_v3_1_gpu.py

# 结构 / 微调 / 消融
./.venv/Scripts/python.exe scripts/run_v3_1_experiments.py \
  --experiments base,L,M,D --seeds 42,43,44 --epochs 100 --patience 18 \
  --batch-size 16 --grad-accum 2 --amp bf16 --out experiments/v3_1/group1
./.venv/Scripts/python.exe scripts/run_v3_1_experiments.py \
  --experiments ft_smiles,ft_both --arch base --seeds 42,43,44 \
  --epochs 100 --patience 18 --batch-size 16 --grad-accum 2 \
  --amp bf16 --out experiments/v3_1/group2

# V3.1 预测（输入 CSV 仍需 smiles、confseq、24 个 PaDEL 列）
molpredict v3-predict \
  --model experiments/v3_1/group2/runs/V3_1_ft_both_seed44 \
  --input new_molecules.csv --out predictions.csv
```

`ft_both` 的 checkpoint 含两个编码器，约 240 MB；加载需要本地 ChemBERTa 缓存和
官方 ConfSeq 仓库。预测输出仍是 PPDB 杀菌剂分类分数，不是实验活性。

预测：

```bash
molpredict v3-predict \
  --model experiments/v3/runs/V3_full_seed42 \
  --input new_molecules.csv \
  --out predictions.csv
```

输入 CSV 必须包含 `smiles`、`confseq` 和 24 个 PaDEL 列；ConfSeq/PaDEL 必须用官方流程
生成，不能用 RDKit 3D 或相似描述符代替。输出是 PPDB 杀菌剂分类分数，不是实验活性概率。

## 外部验证（未参与训练的已知杀菌剂）

从 PubChem 按名称获取结构，按 InChIKey 和连接性（InChIKey 第一段）排除训练集及同连接性分子，得到 8 个已知杀菌剂和 13 个对照（3 个除草剂、2 个杀虫/杀螨剂、8 个普通药物）。使用 `combined` 模型：

| 已知杀菌剂 | 预测分数 |
| --- | --- |
| phthalide（稻瘟病） | 0.870 |
| pyroxyfur（稻瘟病） | 0.721 |
| fluoxapiprolin（OSBPI） | 0.676 |
| ipflufenoquin（DHODHI） | 0.594 |
| pyrimorph（CAA） | 0.526 |
| diclocymet（稻瘟病） | 0.365 |
| metyltetraprole（QoI） | 0.237 |
| isoprothiolane（稻瘟病） | 0.085 |

- 8 个杀菌剂中 5 个 ≥ 0.5，中位数 0.560；
- 13 个对照中 2 个 ≥ 0.5（etoxazole 0.652、acetaminophen 0.592），中位数 0.328；
- 外部集（21 个分子，8 阳性）AUC 0.750、PR-AUC 0.745，低于内部测试集 AUC 0.878；
- 三个模型对比：combined AUC 0.750 > Morgan 0.702 > 描述符 0.625。

同连接性但立体化学/盐型不同的 10 个分子（如 epoxiconazole、fenpicoxamid、mefenoxam）未计入外部统计，因为训练集已有同一母体结构。逐条结果见 `models/external_validation.csv`。

结论：模型对部分未见过的杀菌剂有排序信号，但会漏掉 metyltetraprole、isoprothiolane、diclocymet，也会把个别非杀菌剂排高；样本仅 8 个阳性，**不能作为可靠的实验筛选依据**。

### V3.1 在同一外部集上的补充验证

用官方 ConfSeq 流程和 PaDEL 重新生成输入后，V3.1 在同样 21 个分子上的表现：

| 模型 | 外部 AUC | 杀菌剂 ≥0.5 | 对照 ≥0.5 |
| --- | ---: | ---: | ---: |
| CatBoost combined | **0.750** | 5/8 | 2/13 |
| V3.1 `base` | 0.596 | 4/8 | 4/13 |
| V3.1 `no_smiles` | 0.644 | 4/8 | 4/13 |
| V3.1 `ft_smiles` | 0.462 | 3/8 | 8/13 |
| V3.1 `ft_both` | 0.558 | 4/8 | 7/13 |
| V3.1 `ft_no_smiles` | 0.673 | 5/8 | 7/13 |

- 21 个分子中 11 个与训练集共享 Murcko scaffold，因此外部指标偏乐观；
  只看 10 个 scaffold-disjoint 分子（5 阳性）时，CatBoost AUC 0.520，
  V3.1 全部 ≤0.44，均没有可靠的新骨架外推能力。
- V3.1 的内部测试 AUC（最高 0.8342）没有转化为外部优势；`ft_smiles`/
  `ft_both` 反而把更多对照排到 0.5 以上。
- 逐分子结果：`experiments/v3_1/external/external_predictions.csv`；
  汇总：`experiments/v3_1/external/external_summary.json`。
- **外部验证仍不支持用 V3.1 做实验筛选；CatBoost A 仍是外部表现最好的模型。**

## 限制

- 标签是 PPDB 的**农药分类**（是否为杀菌剂），不是实验测定的活性值；阴性是“已归类为除草剂/杀虫剂”，不是“实验证明无杀菌活性”。
- APPf 原文的 489/6409 原始数据未公开，本项目用 PPDB 重建，规模和分布不同，指标不能与论文直接对比。
- 数据仅 431 个阳性，模型结果有较大不确定性；ConfSeq embedding 在当前数据量下未带来提升。
- V2 只保证标签与 scaffold 划分隔离，未审计 ChemBERTa/ConfSeq 预训练语料是否包含 PPDB 分子；预训练重叠可能使指标偏乐观。
- V2 的 PaDEL 24 中 11 个超大分子被跳过、部分 E-State 缺失，D/F/G 使用训练集中位数填补。
- V2 目前只有实验入口，尚未接入 `molpredict predict`；V3 已有 `v3-predict`。V3.1 已用官方 ConfSeq/PaDEL 重新生成 21 个外部已知分子的输入并补做外部验证，但 V3.1 的外部 AUC（0.462–0.673）低于 CatBoost（0.750），仍不能用于实验筛选。
- V3 的冻结 `full` 没有超过 V2；最好的冻结三划分 V3 配置是 `no_smiles`（0.8066 AUC），仍低于 CatBoost A。`full_ft` 三划分均值 0.8191，与 V2 F-ft 持平，仍低于 A。
- V3 同样未审计预训练语料重叠（状态为 `unknown`）；三套 scaffold 测试集有重叠，均值 ± 标准差不是三次独立复现。
- V3.1 已把 `.venv` 的 torch 从 CPU-only 换成 `2.14.1+cu130`；V3 的旧结果仍是 CPU 训练，不能把 V3/V3.1 的时间或数值差异全部归因于 GPU/AMP。
- V3.1 增加容量没有稳定收益；最好的 0.8342（`ft_both`）仍低于 CatBoost A 0.8578，且部分微调需要 2.3–3.4 倍训练时间。
- V3.1 早停按验证 AUC，而验证损失常在后期的 1.5–3.0 之间恶化；选出的 checkpoint 校准较差，阈值在验证集上不稳定。
- `ft_no_smiles` 是补充实验，不是严格单变量消融（它的 ConfSeq 是微调的，而 `no_smiles` 是冻结的）；不能把 0.003 的差异解释成精确的因果效应。
- V3.1 的外部验证只有 8 个已知杀菌剂；11/21 个外部分子与训练集共享 Murcko scaffold，10 个 scaffold-disjoint 分子的 AUC 全部 ≤0.52（CatBoost）或 ≤0.44（V3.1），样本太小，不能据此估计真实外部性能。
- V3.1 的 `ft_smiles`/`ft_both` 在外部集上把 7–8/13 个对照排到 0.5 以上；内部测试提升不代表外部可用。
- 分子名称与 InChIKey 不作为特征，只用于识别与去重。
