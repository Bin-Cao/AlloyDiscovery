import numpy as np
import torch
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from torch.utils.data import DataLoader

from .data import (
    COMPOSITION_COLS,
    PROCESS_CAT_COLS,
    PROCESS_CONT_COLS,
    TARGET_COLS,
    TEST_COLS,
    TabularDataset,
    validate_feature_columns,
)


RAW_FEATURE_COLS = COMPOSITION_COLS + PROCESS_CAT_COLS + PROCESS_CONT_COLS + TEST_COLS
RESIDUAL_FEATURE_COLS = RAW_FEATURE_COLS


def make_residual_learners(random_state=42):
    return [
        ExtraTreesRegressor(
            n_estimators=500,
            random_state=random_state,
            n_jobs=-1,
            min_samples_leaf=1,
            max_features=1.0,
        ),
        MultiOutputRegressor(
            GradientBoostingRegressor(
                n_estimators=900,
                learning_rate=0.03,
                max_depth=3,
                subsample=0.9,
                random_state=random_state,
            )
        ),
    ]


def _predict_aton_dataset(model, dataset, batch_size=64):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            pred_std = model(batch)
            pred = pred_std * torch.as_tensor(dataset.y_std) + torch.as_tensor(dataset.y_mean)
            preds.append(pred)
    return torch.cat(preds, dim=0).cpu().numpy()


def _blend_residual(adapter, x):
    pred = np.zeros((len(x), len(TARGET_COLS)), dtype=np.float64)
    for weight, learner in zip(adapter["blend_weights"], adapter["learners"]):
        pred += weight * learner.predict(x)
    return pred


def train_aton_residual_adapter(
    model,
    train_df,
    train_dataset,
    random_state=42,
    batch_size=64,
    aton_anchor_weight=(0.02, 0.02),
):
    """
    Train a residual adapter on top of ATON.

    The adapter never creates a separate user-facing prediction. It is an
    internal ATOM output correction branch.
    """
    validate_feature_columns(train_df, require_targets=True)
    residual_dataset = TabularDataset(
        train_df.copy(),
        trajectory_source="labels",
        y_mean=train_dataset.y_mean,
        y_std=train_dataset.y_std,
        proc_cont_mean=train_dataset.proc_cont_mean,
        proc_cont_std=train_dataset.proc_cont_std,
        test_mean=train_dataset.test_mean,
        test_std=train_dataset.test_std,
        state_mean=train_dataset.state_mean,
        state_std=train_dataset.state_std,
        standardize_targets=True,
        require_targets=True,
    )
    aton_pred = _predict_aton_dataset(model, residual_dataset, batch_size=batch_size)
    y_true = residual_dataset.df[TARGET_COLS].values.astype(np.float64)
    anchor = np.asarray(aton_anchor_weight, dtype=np.float64).reshape(1, -1)
    residual_target = y_true - anchor * aton_pred

    x = residual_dataset.df[RESIDUAL_FEATURE_COLS].values
    learners = make_residual_learners(random_state=random_state)
    for learner in learners:
        learner.fit(x, residual_target)

    return {
        "kind": "aton_residual_adapter",
        "learners": learners,
        "blend_weights": [0.5, 0.5],
        "aton_anchor_weight": list(aton_anchor_weight),
        "feature_cols": RESIDUAL_FEATURE_COLS,
        "target_cols": TARGET_COLS,
    }


def apply_aton_residual_adapter(adapter, aton_pred, aton_dataset):
    x = aton_dataset.df[adapter["feature_cols"]].values
    residual = _blend_residual(adapter, x)
    anchor = np.asarray(adapter.get("aton_anchor_weight", [1.0, 1.0]), dtype=np.float64).reshape(1, -1)
    corrected = anchor * aton_pred + residual
    return corrected, residual
