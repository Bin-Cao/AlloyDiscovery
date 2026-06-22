# ATOM Alloy Prediction

ATOM is a trajectory-based alloy property prediction model. It predicts:

- `Strength_MPa`
- `Plasticity_%`

The model uses a neural ATOM backbone plus an internal residual correction branch. Users only see the final prediction columns.

## Main Files

```text
data.xlsx                 Training data
trainer.py                Five-fold cross-validation training entry
inference.py              Label-free inference entry
build_search_space.py     Generate virtual search space
src/data.py               Column definitions and dataset construction
src/model.py              ATOM neural model
src/ensemble.py           Internal residual correction branch
src/train.py              Training and self-consistent prediction utilities
docs/algorithm.html       Detailed English/Chinese algorithm document
docs/figs/plot.ipynb      Five-fold comparison plots
```

## Data Format

Training data must contain these columns:

```text
Al, Co, Cr, Fe, Ni, Ti, Mo, Nb
Eutectic, Preparation, Processing, Tensile/compress
Cold CR_%, Annealing_Temp_K, Annealing_time_h, Aging_Temp_K, Aging_Time_h
Rate_S-1, Tes_Temp_K
Strength_MPa, Plasticity_%
```

Inference data only needs feature columns. If label columns are present, `inference.py` drops them automatically.

## Train

Run from the project root:

```bash
python trainer.py
```

Outputs are generated automatically:

```text
checkpoints/model_best.pth
checkpoints/cv_oof_predictions.xlsx
checkpoints/cv_summary.json
```

The cross-validation result is computed by concatenating all five held-out fold predictions, then evaluating once on the full out-of-fold prediction vector.

Useful options:

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

## Inference

```bash
python inference.py input.xlsx predictions.xlsx --checkpoint checkpoints/model_best.pth
```

Final prediction columns:

```text
ATOM_Predicted_Strength_MPa
ATOM_Predicted_Plasticity_%
```

## Build Search Space

```bash
python build_search_space.py --data data.xlsx --output search_space.csv
python inference.py search_space.csv search_predictions.xlsx --checkpoint checkpoints/model_best.pth
```

## Plot Model Comparison

First train the model:

```bash
python trainer.py
```

Then open and run:

```text
docs/figs/plot.ipynb
```

The notebook compares ATOM with RF, GBDT, SVR, Ridge, Lasso, ElasticNet, and KNN using the same five-fold out-of-fold protocol.

## Use Your Own Data

1. Prepare an Excel or CSV file with the required feature columns.
2. Keep `Strength_MPa` and `Plasticity_%` only for training data.
3. Encode categorical columns as non-negative integers.
4. If your element or process columns are different, edit the column lists in `src/data.py`.
5. Train with:

```bash
python trainer.py --data your_data.xlsx
```

For detailed algorithm notes, open:

```text
docs/algorithm.html
```
