# N/A 条件说明

计划矩阵中的以下条件**没有实施**，原因不是资源，而是设计上无法公平实施：

| 条件 | 状态 | 原因 |
| --- | --- | --- |
| `A_NEW_AUX` | N/A | A 只有 SMILES 输入，没有辅助任务；把新数据加入主任务会改变主任务的数据或标签语义 |
| `B_NEW_AUX` | N/A | B 只有 24 PaDEL 输入，无法从结构做辅助任务；新数据没有可用的旧任务标签 |
| `C_NEW_AUX` | N/A | C 没有辅助头；加入新数据只能变成 D，或改变主任务数据 |
| `E_NEW_AUX` | N/A | 同 C；深度实验不包含辅助任务 |

可实施的新数据条件只有 D：

- `D_OLD_ONLY`：辅助任务只用旧训练集（不含旧 val/test）。
- `D_NEW_AUX_MAIN`：辅助任务 = 旧训练集 + 严格隔离的新 MAIN（45 个结构）。
- `D_NEW_AUX_FULL`：辅助任务 = 旧训练集 + 严格隔离的 MAIN+EXTENDED（199 个结构）。

严格隔离 = 排除旧项目任何 InChIKey、重复结构、MULTI_COMPONENT，以及
Murcko scaffold 与旧 val/test 重叠的新结构。详见
`INITIAL_AUDIT.md` 和 `data_audit/new_aux_audit.json`。

另外，新数据只有正标签/UNKNOWN（明确负样本 0），所以不能把 FUNGICIDE 等
用途标签转成旧任务标签，也不能用 UNKNOWN 当负样本；因此没有实施用途标签
辅助任务。辅助任务采用结构→24 PaDEL。
