<div align="center">

# [ATON for Alloy Discovery](https://bin-cao.github.io/AlloyDiscovery/)

**自己無撞着推論と残差補正を組み合わせた、軌跡ベースの合金物性予測。**

[English](../README.md) | [中文](README.zh-CN.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Español](README.es.md) | [Deutsch](README.de.md)

</div>

Alloy Trajectory Optimization Network/Model (ATON/ATOM) は、組成およびプロセス記述子から合金の機械的性質を予測します。

- `Strength_MPa`
- `Plasticity_%`

このモデルは、ATOM ニューラルバックボーン、自己無撞着な軌跡推論、内部残差補正ブランチを組み合わせています。ユーザー向けの予測ファイルには、最終的な ATOM 予測列のみが出力されます。

## 結果プレビュー

<p align="center">
  <img src="figs/tensile.png" alt="強度予測の比較" width="48%">
  <img src="figs/elongation.png" alt="延性予測の比較" width="48%">
</p>

比較用 notebook は、同じ 5-fold out-of-fold プロトコルで ATOM と RF、GBDT、SVR、Ridge、Lasso、ElasticNet、KNN を評価します。

## クイックスタート

```bash
python trainer.py
python inference.py input.xlsx predictions.xlsx --checkpoint checkpoints/model_best.pth
```

生成される学習成果物：

```text
checkpoints/model_best.pth
checkpoints/cv_oof_predictions.xlsx
checkpoints/cv_summary.json
```

最終推論列：

```text
ATOM_Predicted_Strength_MPa
ATOM_Predicted_Plasticity_%
```

## リポジトリ構成

```text
data.xlsx                 学習データ
trainer.py                5-fold クロスバリデーション学習入口
inference.py              ラベルなし推論入口
build_search_space.py     仮想探索空間の生成
src/data.py               列定義とデータセット構築
src/model.py              ATOM ニューラルモデル
src/ensemble.py           内部残差補正ブランチ
src/train.py              学習と自己無撞着予測のユーティリティ
docs/algorithm.html       詳細な英語/中国語アルゴリズム文書
docs/README.md            多言語 README 切り替え入口
docs/figs/plot.ipynb      5-fold 比較プロット
```

## データ形式

学習データには次の列が必要です。

```text
Al, Co, Cr, Fe, Ni, Ti, Mo, Nb
Eutectic, Preparation, Processing, Tensile/compress
Cold CR_%, Annealing_Temp_K, Annealing_time_h, Aging_Temp_K, Aging_Time_h
Rate_S-1, Tes_Temp_K
Strength_MPa, Plasticity_%
```

推論データには特徴量列のみが必要です。ラベル列が含まれている場合、`inference.py` が自動的に削除します。

## 学習

プロジェクトルートで実行します。

```bash
python trainer.py
```

クロスバリデーション結果は、5 つの held-out fold 予測を連結し、完全な out-of-fold 予測ベクトル上で一度だけ評価して計算されます。

主なオプション：

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

## 推論

```bash
python inference.py input.xlsx predictions.xlsx --checkpoint checkpoints/model_best.pth
```

## 探索空間

```bash
python build_search_space.py --data data.xlsx --output search_space.csv
python inference.py search_space.csv search_predictions.xlsx --checkpoint checkpoints/model_best.pth
```

## モデル比較プロット

まずモデルを学習します。

```bash
python trainer.py
```

次に開いて実行します。

```text
docs/figs/plot.ipynb
```

## 独自データの使用

1. 必要な特徴量列を含む Excel または CSV ファイルを用意します。
2. `Strength_MPa` と `Plasticity_%` は学習データにのみ残します。
3. カテゴリ列を非負整数としてエンコードします。
4. 元素列またはプロセス列が異なる場合は、`src/data.py` の列リストを編集します。
5. 学習を実行します。

```bash
python trainer.py --data your_data.xlsx
```

## ドキュメント

完全なアルゴリズムガイド：

- [オンライン文書](https://bin-cao.github.io/AlloyDiscovery/)
- [`docs/algorithm.html`](algorithm.html)
