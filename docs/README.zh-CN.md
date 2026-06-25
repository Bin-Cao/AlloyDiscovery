<div align="center">

# [ATON for Alloy Discovery](https://bin-cao.github.io/AlloyDiscovery/)

**基于轨迹的合金性能预测，结合自洽推理与残差校正。**

[English](../README.md) | [中文](README.zh-CN.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Español](README.es.md) | [Deutsch](README.de.md)

</div>

Alloy Trajectory Optimization Network/Model (ATON/ATOM) 根据成分和工艺描述符预测合金力学性能：

- `Strength_MPa`
- `Plasticity_%`

模型结合 ATOM 神经网络骨干、自洽轨迹推理和内部残差校正分支。面向用户的预测文件只输出最终 ATOM 预测列。

## 结果预览

<p align="center">
  <img src="figs/tensile.png" alt="强度预测对比" width="48%">
  <img src="figs/elongation.png" alt="塑性预测对比" width="48%">
</p>

对比 notebook 在相同五折 out-of-fold 流程下评估 ATOM 与 RF、GBDT、SVR、Ridge、Lasso、ElasticNet 和 KNN。

## 快速开始

```bash
python trainer.py
python inference.py input.xlsx predictions.xlsx --checkpoint checkpoints/model_best.pth
```

训练生成的文件：

```text
checkpoints/model_best.pth
checkpoints/cv_oof_predictions.xlsx
checkpoints/cv_summary.json
```

最终推理输出列：

```text
ATOM_Predicted_Strength_MPa
ATOM_Predicted_Plasticity_%
```

## 仓库结构

```text
data.xlsx                 训练数据
trainer.py                五折交叉验证训练入口
inference.py              无标签推理入口
build_search_space.py     生成虚拟搜索空间
src/data.py               列定义和数据集构建
src/model.py              ATOM 神经网络模型
src/ensemble.py           内部残差校正分支
src/train.py              训练和自洽预测工具
docs/algorithm.html       详细英文/中文算法文档
docs/README.md            多语言 README 切换入口
docs/figs/plot.ipynb      五折对比图
```

## 数据格式

训练数据必须包含以下列：

```text
Al, Co, Cr, Fe, Ni, Ti, Mo, Nb
Eutectic, Preparation, Processing, Tensile/compress
Cold CR_%, Annealing_Temp_K, Annealing_time_h, Aging_Temp_K, Aging_Time_h
Rate_S-1, Tes_Temp_K
Strength_MPa, Plasticity_%
```

推理数据只需要特征列。如果包含标签列，`inference.py` 会自动删除。

## 训练

在项目根目录运行：

```bash
python trainer.py
```

交叉验证结果通过拼接五个 held-out fold 的预测后，在完整 out-of-fold 预测向量上统一计算。

常用参数：

```bash
python trainer.py \
  --data data.xlsx \
  --output-dir checkpoints \
  --epochs 300 \
  --batch-size 32 \
  --lr 1e-3 \
  --patience 50 \
  --sc-max-iters 30 \
  --sc-tol 1e-3
```

## 推理

```bash
python inference.py input.xlsx predictions.xlsx --checkpoint checkpoints/model_best.pth
```

## 搜索空间

```bash
python build_search_space.py --data data.xlsx --output search_space.csv
python inference.py search_space.csv search_predictions.xlsx --checkpoint checkpoints/model_best.pth
```

## 模型对比图

先训练模型：

```bash
python trainer.py
```

然后打开并运行：

```text
docs/figs/plot.ipynb
```

## 使用自己的数据

1. 准备包含必需特征列的 Excel 或 CSV 文件。
2. 只在训练数据中保留 `Strength_MPa` 和 `Plasticity_%`。
3. 将类别列编码为非负整数。
4. 如果元素列或工艺列不同，修改 `src/data.py` 中的列列表。
5. 训练：

```bash
python trainer.py --data your_data.xlsx
```

## 文档

完整算法文档：

- [在线文档](https://bin-cao.github.io/AlloyDiscovery/)
- [`docs/algorithm.html`](algorithm.html)
