import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold

from src.data import (
    COMPOSITION_COLS,
    PROCESS_CONT_COLS,
    TARGET_COLS,
    TEST_COLS,
    TabularDataset,
    get_proc_cat_dims,
    read_alloy_data,
    validate_feature_columns,
)
from src.ensemble import apply_aton_residual_adapter, train_aton_residual_adapter
from src.model import TrajectoryTabularModel
from src.train import compute_metrics, predict_self_consistent, print_metrics, train_model


CONFIG = {
    "data_path": "data.xlsx",
    "sheet_name": 0,
    "output_dir": "checkpoints",
    "n_splits": 5,
    "random_state": 42,
    "epochs": 300,
    "batch_size": 32,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "early_stopping_patience": 50,
    "task_weights": [1.0, 1.0],
    "emb_dim": 20,
    "proc_out_dim": 96,
    "comp_hidden": 192,
    "dropout": 0.1,
    "sc_max_iters": 30,
    "sc_tol": 1e-3,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train ATON with 5-fold out-of-fold self-consistent CV."
    )
    parser.add_argument("--data", default=CONFIG["data_path"])
    parser.add_argument("--sheet", default=CONFIG["sheet_name"])
    parser.add_argument("--output-dir", default=CONFIG["output_dir"])
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    parser.add_argument("--batch-size", type=int, default=CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=CONFIG["lr"])
    parser.add_argument("--patience", type=int, default=CONFIG["early_stopping_patience"])
    parser.add_argument("--sc-max-iters", type=int, default=CONFIG["sc_max_iters"])
    parser.add_argument("--sc-tol", type=float, default=CONFIG["sc_tol"])
    return parser.parse_args()


def normalize_sheet_name(sheet):
    if isinstance(sheet, str) and sheet.isdigit():
        return int(sheet)
    return sheet


def build_model(df, config):
    return TrajectoryTabularModel(
        comp_dim=len(COMPOSITION_COLS),
        proc_cat_dims=get_proc_cat_dims(df),
        proc_cont_dim=len(PROCESS_CONT_COLS),
        test_dim=len(TEST_COLS),
        emb_dim=config["emb_dim"],
        proc_out_dim=config["proc_out_dim"],
        comp_hidden=config["comp_hidden"],
        dropout=config["dropout"],
    )


def main():
    args = parse_args()
    config = CONFIG.copy()
    config.update(
        {
            "data_path": args.data,
            "sheet_name": normalize_sheet_name(args.sheet),
            "output_dir": args.output_dir,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "early_stopping_patience": args.patience,
            "sc_max_iters": args.sc_max_iters,
            "sc_tol": args.sc_tol,
        }
    )

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {config['data_path']}...")
    df = read_alloy_data(config["data_path"], sheet_name=config["sheet_name"])
    validate_feature_columns(df, require_targets=True)
    df = df.reset_index(drop=True)
    df["_original_index"] = np.arange(len(df))
    print(f"Loaded {len(df)} rows")

    y_true_all = df[TARGET_COLS].values.astype(np.float32)
    oof_pred = np.zeros_like(y_true_all, dtype=np.float32)
    fold_records = []

    splitter = KFold(
        n_splits=config["n_splits"],
        shuffle=True,
        random_state=config["random_state"],
    )

    best_fold_loss = float("inf")
    best_fold_checkpoint = None

    for fold, (train_idx, val_idx) in enumerate(splitter.split(df), start=1):
        print("\n" + "=" * 80)
        print(f"Fold {fold}/{config['n_splits']}")
        print("=" * 80)

        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()

        train_dataset = TabularDataset(
            train_df,
            trajectory_source="labels",
            standardize_targets=True,
            require_targets=True,
        )

        model = build_model(df, config)
        fold_checkpoint = output_dir / f"model_fold{fold}_best.pth"

        checkpoint_extra = {
            "fold": fold,
            "config": config,
            "y_mean": train_dataset.y_mean.tolist(),
            "y_std": train_dataset.y_std.tolist(),
            "proc_cont_mean": train_dataset.proc_cont_mean.tolist(),
            "proc_cont_std": train_dataset.proc_cont_std.tolist(),
            "test_mean": train_dataset.test_mean.tolist(),
            "test_std": train_dataset.test_std.tolist(),
            "state_mean": train_dataset.state_mean.tolist(),
            "state_std": train_dataset.state_std.tolist(),
            "proc_cat_dims": get_proc_cat_dims(df),
            "model_config": {
                "comp_dim": len(COMPOSITION_COLS),
                "proc_cat_dims": get_proc_cat_dims(df),
                "proc_cont_dim": len(PROCESS_CONT_COLS),
                "test_dim": len(TEST_COLS),
                "emb_dim": config["emb_dim"],
                "proc_out_dim": config["proc_out_dim"],
                "comp_hidden": config["comp_hidden"],
                "dropout": config["dropout"],
            },
        }

        fold_loss, best_epoch = train_model(
            model,
            train_dataset,
            val_df=val_df,
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            lr=config["lr"],
            save_path=fold_checkpoint,
            save_every=0,
            early_stopping_patience=config["early_stopping_patience"],
            weight_decay=config["weight_decay"],
            task_weights=config["task_weights"],
            sc_max_iters=config["sc_max_iters"],
            sc_tol=config["sc_tol"],
            checkpoint_extra=checkpoint_extra,
        )

        checkpoint = torch.load(fold_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        residual_adapter = train_aton_residual_adapter(
            model,
            train_df,
            train_dataset,
            random_state=config["random_state"],
            batch_size=config["batch_size"] * 2,
        )
        checkpoint["residual_adapter"] = residual_adapter
        checkpoint["checkpoint_type"] = "aton_with_residual_adapter"
        torch.save(checkpoint, fold_checkpoint)

        fold_pred_dataset_order, fold_dataset = predict_self_consistent(
            model,
            val_df,
            train_dataset.y_mean,
            train_dataset.y_std,
            proc_cont_mean=train_dataset.proc_cont_mean,
            proc_cont_std=train_dataset.proc_cont_std,
            test_mean=train_dataset.test_mean,
            test_std=train_dataset.test_std,
            state_mean=train_dataset.state_mean,
            state_std=train_dataset.state_std,
            max_iters=config["sc_max_iters"],
            tol=config["sc_tol"],
            batch_size=config["batch_size"] * 2,
            verbose=False,
        )
        fold_corrected_pred, fold_residual = apply_aton_residual_adapter(
            residual_adapter,
            fold_pred_dataset_order,
            fold_dataset,
        )
        val_positions = fold_dataset.df["_original_index"].values.astype(int)
        oof_pred[val_positions] = fold_corrected_pred

        fold_records.append(
            {
                "fold": fold,
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
                "best_epoch": int(best_epoch),
                "best_self_consistent_val_loss": float(fold_loss),
                "checkpoint": str(fold_checkpoint),
            }
        )

        if fold_loss < best_fold_loss:
            best_fold_loss = fold_loss
            best_fold_checkpoint = fold_checkpoint

    metrics = compute_metrics(y_true_all, oof_pred)
    print("\n" + "=" * 80)
    print("ATOM 5-fold out-of-fold performance")
    print("=" * 80)
    print_metrics(metrics)

    oof_df = df.drop(columns=["_original_index"]).copy()
    oof_df["ATOM_OOF_Predicted_Strength_MPa"] = oof_pred[:, 0]
    oof_df["ATOM_OOF_Predicted_Plasticity_%"] = oof_pred[:, 1]
    oof_df["ATOM_OOF_Strength_Error"] = oof_df["ATOM_OOF_Predicted_Strength_MPa"] - oof_df[TARGET_COLS[0]]
    oof_df["ATOM_OOF_Plasticity_Error"] = oof_df["ATOM_OOF_Predicted_Plasticity_%"] - oof_df[TARGET_COLS[1]]
    oof_path = output_dir / "cv_oof_predictions.xlsx"
    oof_df.to_excel(oof_path, index=False)

    summary = {
        "oof_metrics": metrics,
        "folds": fold_records,
        "best_fold_checkpoint": str(best_fold_checkpoint),
        "best_fold_loss": best_fold_loss,
    }
    summary_path = output_dir / "cv_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    final_checkpoint = output_dir / "model_best.pth"
    if best_fold_checkpoint is not None:
        shutil.copy2(best_fold_checkpoint, final_checkpoint)

    print(f"\nSaved OOF predictions to {oof_path}")
    print(f"Saved CV summary to {summary_path}")
    print(f"Saved best fold checkpoint to {final_checkpoint}")


if __name__ == "__main__":
    main()
