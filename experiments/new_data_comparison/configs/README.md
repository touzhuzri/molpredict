# 配置说明

实验名 → 输入/条件：

- `A_OLD_ONLY`：SMILES only（无 ConfSeq、无 PaDEL）。
- `B_OLD_ONLY`：24 PaDEL only。
- `C_OLD_ONLY`：SMILES + 24 PaDEL。
- `D_OLD_ONLY`：C + 辅助任务；辅助集 = 旧训练集（不含旧 val/test）。
- `D_NEW_AUX_MAIN`：C + 辅助任务；辅助集 = 旧训练集 + 严格隔离的新 MAIN（45）。
- `D_NEW_AUX_FULL`：C + 辅助任务；辅助集 = 旧训练集 + 严格隔离的 MAIN+EXTENDED（199）。
- `E_OLD_ONLY`：C 的输入，8 层 Transformer（C 为 4 层）。

A/B/C/E 的 NEW_AUX 条件为 N/A；见 `../metrics/na_experiments.md`。

所有配置：
- 128 维、4 头（E 为 8 层）、FFN 2×、dropout 0.3、CLS pooling；
- AdamW，lr=3e-4，weight_decay=1e-4，batch 16×2 梯度累积；
- warmup 10% + cosine，最多 100 epoch，验证 ROC-AUC 早停 patience=18；
- BF16，grad clip 5.0；阈值验证集最大 F1；测试集只评估一次；
- 主任务数据/划分固定为旧项目 seed 42/43/44。

每个 YAML 由 `scripts/run_new_data_experiments.py` 在运行前写入；每个 run 目录
还有 `config.json`（含实际参数量、设备、代码哈希、数据哈希）。
