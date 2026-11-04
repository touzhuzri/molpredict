# molpredict V2：统一多模态杀菌剂分类模型

## 旧架构（M3，保留）

```text
SMILES ──RDKit──┬─ Morgan 指纹 2048
                ├─ RDKit 描述符 217
                └─（可选）ConfSeq 256 维 embedding（已预先算好）
                         │
                  拼接特征向量
                         │
                 CatBoost 二分类
                         │
                 杀菌剂分类分数
```

旧架构是“特征工程 + 树模型”，每种特征组合单独训练一个 CatBoost；没有可学习的融合层，
SMILES 本身不经过预训练 Transformer。

## 新架构（V2）

```text
                    SMILES 字符串
                         │
              ChemBERTa BPE tokenizer
                         │
          ChemBERTa 编码器（冻结，可部分解冻）
                         │  token states [L_s, 768]
                 线性投影 → LayerNorm → GELU
                         │
                  可学习注意力池化
                         │  z_s (128)
                         │
┌────────────────────────┼────────────────────────┐
│                        │                        │
ConfSeq 字符串      数值描述符                     │
    │                    │                        │
官方 ConfSeq BART    PaDEL 24 / RDKit 217         │
（冻结）              （train-only 标准化）         │
    │                    │                        │
token states          MLP 投影                     │
[L_c, 256]               │                        │
    │                    │                        │
投影 + 注意力池化       z_n (128)                   │
    │                    │                        │
    z_c (128)            │                        │
    └────────────┬───────┘                        │
                 ▼                                ▼
        融合层（z_s, z_c, z_n）
        ├─ 门控融合：softmax(MLP(concat(z))) 加权求和
        └─ 注意力融合：模态 token + type embedding
                        + CLS → 1 层 Transformer → z
                         │
                  分类头 (128→64→1)
                         │
                 PPDB 杀菌剂分类概率
```

## 各实验输入

| 实验 | 输入 |
| --- | --- |
| A | Morgan 2048 + RDKit 217 描述符，CatBoost |
| B | SMILES（ChemBERTa token states） |
| C | ConfSeq（官方编码器 token states） |
| D | PaDEL 24 描述符 |
| E | SMILES + ConfSeq |
| F | SMILES + ConfSeq + PaDEL 24 |
| G | SMILES + ConfSeq + RDKit 217 |

## 预训练权重

| 分支 | 模型 | 来源 | 许可证 | 验证 |
| --- | --- | --- | --- | --- |
| SMILES | ChemBERTa `seyonec/ChemBERTa-zinc-base-v1` | Hugging Face（HF 镜像下载） | 代码仓库 MIT | 实际加载：44,103,936 参数，768 维，6 层，tokenizer 767 词表 |
| ConfSeq | 官方 `model_epoch_1.pth` | HF 镜像 `Oopstom/confseq-shape-screen` | ConfSeq 仓库 AGPL-3.0 | 官方 `build_bart_encoder` + `strict=True` 加载成功；14,838,016 参数，256 维，6 层 |
| 数值 | PaDEL-Descriptor | padelpy 0.1.17 内置 jar | jar AGPL-3.0 | Java 21 下实际计算，24 个名称逐项匹配 |

说明：官方 ConfSeq OneDrive 共享链接匿名访问返回 401，因此使用 HF 镜像中的同名 checkpoint；未修改 ConfSeq 算法。ChemBERTa 的 HF 模型卡未单独标注许可证，上游代码仓库为 MIT；权重仅本地研究使用。

PaDEL 覆盖：1823 个分子中 1812 个 24 项完整；11 个 >60 重原子的聚合/复杂分子（Chitosan hydrochloride、Streptomycin sulphate 等）会卡住 PaDEL，被跳过；另有 11 个分子的部分 E-State 列为空，`VR1_DzZ` 对 58 个对称/特殊结构未定义。缺失值只用训练集对应描述符的中位数填补，并在 `padel24_meta.json` 中逐条记录。未用 RDKit 同名特征替代。

环境：机器有 RTX 4060 8GB，但项目当前为 CPU 版 PyTorch；使用冻结编码器 + 预先缓存 token 表示，不安装数 GB 的 CUDA 依赖。

## 数据与隔离

- 统一使用 `data/training/confseq.csv` 的 1823 个分子（431 阳性 / 1392 阴性）。
- Murcko scaffold 分组划分：测试集 20%；训练集内部再按 scaffold 划 15% 验证集。
- 使用种子 42/43/44 生成三套 scaffold 划分，报告 mean±std。
- 数值标准化（中位数填补、均值/方差）只用训练集拟合。
- 阈值只在验证集上按最大 F1 选择，测试集只评估一次。
- 预训练编码器在初始实验中冻结；额外做 ChemBERTa 最后 2 层部分解冻实验。

## 运行

```bash
# 1. 生成缓存（PaDEL、RDKit、ChemBERTa token states、ConfSeq token states）
./.venv/Scripts/python.exe scripts/run_v2_experiments.py --prepare

# 2. 运行 A–G，三套 scaffold 划分，E/F/G 同时比较门控/注意力融合
./.venv/Scripts/python.exe scripts/run_v2_experiments.py \
  --experiments A,B,C,D,E,F,G --seeds 42,43,44 \
  --fusions attention,gated --epochs 60 --patience 12

# 3. 额外：F 的 ChemBERTa 最后 2 层部分解冻（seed 42）
./.venv/Scripts/python.exe scripts/run_v2_experiments.py \
  --experiments F --seeds 42 --fusions attention --finetune
```

输出：`experiments/v2/runs/<实验>_<融合>_seed<N>/`（config、model.pt、predictions_test.csv、metrics.json）；汇总在 `experiments/v2/summary.csv` 和 `stability.json`。

## 指标

ROC-AUC、PR-AUC、F1、Precision、Recall、MCC、混淆矩阵（TN/FP/FN/TP）。

## 限制

- 标签是 PPDB 农药分类，不是实验杀菌活性。
- 1823 个分子、431 个阳性，外部泛化能力有限。
- ConfSeq 分支使用官方编码器 token 表示，不修改其编码算法；主实验冻结，只对 ChemBERTa 做部分解冻。
- PaDEL/ConfSeq 组件为 AGPL-3.0；本项目仅在本地研究使用，不重新分发。
- PaDEL 的 11 个超大分子缺失、11 个分子部分 E-State 缺失，训练集中位数填补；这可能降低 D/F/G 在这些分子上的表现。
- 只保证标签与 scaffold 划分隔离，未审计 ChemBERTa/ConfSeq 预训练语料是否包含 PPDB 分子；预训练重叠可能使指标偏乐观。
