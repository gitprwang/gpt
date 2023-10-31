# 数据

## 原始数据
数据存在`/home/data/wd`目录下，原始数据分别为`rxn.txt`，`zinc_data.txt`和`USPTO/uspto_data.txt`。其中，`rxn.txt`和`USPTO/uspto_data.txt`为反应数据，每行对应一条反应数据，格式为
```
reactant1.reactant2>catalyst1.solvent1>product1.product2
```
`zinc_data.txt`为分子式数据，每行对应一个分子的SMILES。

## 数据处理
`zinc`数据集用于构建性质预测和分子生成的数据。
格式如下：
```
props_embedding <tok> scaffold>target <tok> props_embedding
```

反应数据组织为如下格式：
```
null <tok> reactant1.reactant2>catalyst1.solvent1>product1.product2 <tok> props_embedding_of_product2
```

## dataloader
- [ ] 四个任务的数据比例
- [ ] tokenizer在dataset里做还是model里做