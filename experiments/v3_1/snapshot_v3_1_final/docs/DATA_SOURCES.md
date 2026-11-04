
## V3 使用的数据与隔离

V3 沿用上述 PPDB 重建数据和标签定义，不新增、不修改标签：

- `data/training/confseq.csv`：1823 条（431 阳性 / 1392 阴性），因 4 条 ConfSeq 转换失败
  从 1827 条缩减。
- V2 的 ConfSeq token 缓存和 PaDEL 24 缓存被复用；V3 只重新生成 SMILES token 缓存，
  把 ChemBERTa 最大长度从 128 提高到 320（实测最大 298，未截断）。
- scaffold 划分复用 V2 的 seed 42/43/44，索引复制到 `experiments/v3/splits/`。
- `experiments/v3/leakage_audit.json` 记录：
  - InChIKey 重复、canonical SMILES 重复；
  - train/val/test 之间索引、InChIKey、Murcko scaffold 的重叠；
  - 预训练语料重叠状态。
- 预训练语料（ChemBERTa 的 ZINC、ConfSeq 的预训练语料）没有完整分子清单，因此
  **无法完成逐分子审计**；报告只能标记为 `unknown`，不能声称没有重叠。

## V3.1 使用的数据

V3.1 不新增数据集，也不重新划分：

- 复用 `experiments/v3/cache/` 的 SMILES 320-token 表示、V2 的 ConfSeq token
  表示和 PaDEL 24 缓存；
- 复用 V2 的 scaffold 划分 seed 42/43/44，索引复制到
  `experiments/v3_1/group1/splits/`（group2/group3/smiles_check 使用同一划分）；
- 标签仍是 PPDB 农药分类（431 阳性 / 1392 阴性），不是实验活性；
- 标准化/缺失值中位数仍只在训练集拟合；
- 泄漏审计结果与 V3 相同（InChIKey/canonical SMILES 重复 0，划分重叠 0）；
  预训练语料重叠仍为 `unknown`。
- PyTorch CUDA wheel 属于软件环境，不改变训练数据；安装来源记录在
  `docs/v3_1_report.md` 第 2 节。

## V3.1 外部验证输入

`experiments/v3_1/external/` 的 21 个分子来自 `models/external_validation.csv`，
不在训练集（InChIKey 重叠 0）。为了用 V3.1 预测，重新生成了：

- 官方 ConfSeq 字符串：`scripts/confseq_pipeline.py`（调用 ConfSeq 官方 demo），
  21/21 成功，所有 token 都在词表内；
- 24 个 PaDEL 描述符：`padelpy` 2D 计算，21/21 成功，无 NaN；
- 文件：`external_input.csv`、`external_confseq.csv`、`padel_cache/padel24.csv`、
  `external_features.csv`、`external_predictions.csv`、`external_summary.json`。

这些分子中有 11 个与训练集共享 Murcko scaffold，因此不是干净的 scaffold 外部集；
报告中同时给出了 10 个 scaffold-disjoint 分子的敏感性分析。标签仍来自 PPDB 分类，
不是实验活性；8 个阳性不足以做可靠的泛化结论。
