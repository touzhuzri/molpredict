
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
