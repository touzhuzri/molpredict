# 真实杀菌活性数据申请模板

用途：把当前“杀菌剂-like 原理样机”推进为“可验证的杀菌活性模型”。  
原则：同一化合物、同一病原、同一实验条件；失败/无活性化合物也必须保留。

## 1. 最低可用数据量

| 项目 | 最低 | 建议 |
| --- | ---: | ---: |
| 化合物总数 | 300 | 500–1,000 |
| 病原数 | 3 | 3–5 |
| 明确无活性/弱活性化合物 | 100 | 200 |
| 盲测外部化合物 | 100 | 150–200 |
| 每化合物/病原重复 | 3 | 3–4 |
| 浓度点 | 5 | 6–8（含 0 对照） |

建议病原：

- *Botrytis cinerea*（灰霉）
- *Fusarium graminearum*（赤霉）
- *Magnaporthe oryzae*（稻瘟）

如果实验室有更常做的面板，以实验室实际面板为准；关键是**所有化合物在同一
面板、同一条件下测试**。

## 2. 化合物清单字段

```text
compound_id, compound_name, cas, smiles, inchikey, source,
purity, stock_concentration, solvent, batch, notes
```

要求：

- `smiles` 或 `inchikey` 至少有一个；
- 多成分产品必须拆成组分并标明配方，不得把产品活性直接归给单个组分；
- 已知无活性/弱活性化合物也要有 `compound_id`；
- 盲测集单独标记 `split=external_blind`，不要提前给模型训练。

## 3. 实验记录字段

```text
assay_id, compound_id, pathogen_species, strain, medium, temperature,
inoculum_density, compound_concentration, unit, replicate,
inhibition_percent, mic, ec50, assay_date, operator, plate_id,
positive_control, negative_control, notes
```

要求：

- 记录原始抑制率，不要只给“有/无活性”；
- 记录浓度单位（µg/mL、µM 等）；
- 记录阳性/阴性对照；
- 记录失败/污染/复孔异常，不要删除；
- 同一化合物在不同病原上分别一行。

## 4. 可直接使用的 CSV 模板

```csv
compound_id,compound_name,smiles,pathogen_species,strain,concentration,unit,replicate,inhibition_percent,mic,ec50,assay_date,operator,notes
C001,example,CC(=O)O,Botrytis cinerea,strain-1,10,ug/mL,1,45,,,2026-11-04,tech,example
```

## 5. 人力

| 角色 | 投入 | 工作 |
| --- | --- | --- |
| 数据整理/标注 | 0.5–1 人 × 3 个月 | 结构标准化、去重、标签复核、冲突处理 |
| 微生物实验 | 0.5 人 × 3–6 个月 | 同一面板的 dose-response |
| 导师/植保专家 | 0.1–0.2 人 | 定病原、终点、活性阈值和标签规则 |
| 计算 | 现有 8 GB GPU | demo 和正式训练 |

## 6. 建议的验证设计

1. 用当前 demo 对化合物库排序；
2. 选 Top 10–20% 作为“模型组”，再随机选同样数量的“对照组”；
3. 两组在同一病原面板、同一条件下测试；
4. 比较命中率、抑制率分布和 Top-k 富集；
5. 把结果分为训练/验证/盲测，重新训练后再做一次独立盲测。

## 7. 交付物

- 一份化合物清单（含 SMILES/InChIKey）；
- 一份实验记录（含原始抑制率和对照）；
- 一份盲测集（不参与训练）；
- 一份标签规则说明（什么算正、什么算负、UNKNOWN 怎么处理）。

没有这些字段，模型只能做“用途证据排序”，不能做“活性预测”。
