import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data import PRED_TARGET_COLS, TARGET_COLS, read_alloy_data, validate_feature_columns
from src.ensemble import apply_aton_residual_adapter
from src.model import TrajectoryTabularModel
from src.train import predict_self_consistent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run label-free self-consistent inference with a trained checkpoint."
    )
    parser.add_argument("input", help="Input .xlsx/.csv dataset")
    parser.add_argument("output", help="Output .xlsx/.csv predictions file")
    parser.add_argument("--checkpoint", default="checkpoints/model_best.pth")
    parser.add_argument("--sheet", default=0)
    parser.add_argument("--max-iters", type=int, default=30)
    parser.add_argument("--sc-tol", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--mc-samples", type=int, default=0)
    return parser.parse_args()


def normalize_sheet_name(sheet):
    if isinstance(sheet, str) and sheet.isdigit():
        return int(sheet)
    return sheet


def load_model_from_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_config" not in checkpoint:
        raise ValueError(
            "Checkpoint is missing model_config. Train a new checkpoint with trainer.py."
        )
    if "y_mean" not in checkpoint or "y_std" not in checkpoint:
        raise ValueError(
            "Checkpoint is missing y_mean/y_std. Train a new checkpoint with trainer.py."
        )

    model = TrajectoryTabularModel(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    y_mean = np.asarray(checkpoint["y_mean"], dtype=np.float32)
    y_std = np.asarray(checkpoint["y_std"], dtype=np.float32)
    input_stats = {
        "proc_cont_mean": np.asarray(checkpoint.get("proc_cont_mean"), dtype=np.float32),
        "proc_cont_std": np.asarray(checkpoint.get("proc_cont_std"), dtype=np.float32),
        "test_mean": np.asarray(checkpoint.get("test_mean"), dtype=np.float32),
        "test_std": np.asarray(checkpoint.get("test_std"), dtype=np.float32),
        "state_mean": np.asarray(checkpoint.get("state_mean"), dtype=np.float32),
        "state_std": np.asarray(checkpoint.get("state_std"), dtype=np.float32),
    }
    if any(value.shape == () for value in input_stats.values()):
        raise ValueError(
            "Checkpoint is missing input standardization statistics. "
            "Train a new checkpoint with trainer.py."
        )
    return model, y_mean, y_std, input_stats, checkpoint


def enable_dropout(model):
    for module in model.modules():
        if module.__class__.__name__.startswith("Dropout"):
            module.train()


def predict_mc(model, dataset, y_mean, y_std, n_samples, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    enable_dropout(model)

    samples = []
    with torch.no_grad():
        for _ in range(n_samples):
            batch_preds = []
            for batch in loader:
                pred_std = model(batch)
                pred = pred_std * torch.as_tensor(y_std) + torch.as_tensor(y_mean)
                batch_preds.append(pred)
            samples.append(torch.cat(batch_preds, dim=0))

    stacked = torch.stack(samples, dim=0)
    return stacked.mean(dim=0).numpy(), stacked.std(dim=0).numpy()


def write_output(df, output_path):
    output_path = Path(output_path)
    if output_path.suffix.lower() == ".csv":
        df.to_csv(output_path, index=False)
    else:
        df.to_excel(output_path, index=False)


def main():
    args = parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    model, y_mean, y_std, input_stats, checkpoint = load_model_from_checkpoint(args.checkpoint)
    print(f"Loaded checkpoint epoch={checkpoint.get('epoch')} fold={checkpoint.get('fold')}")

    print(f"Loading input data: {args.input}")
    df_input = read_alloy_data(args.input, sheet_name=normalize_sheet_name(args.sheet))
    validate_feature_columns(df_input, require_targets=False)

    leak_cols = [col for col in TARGET_COLS if col in df_input.columns]
    if leak_cols:
        print(f"Dropping target columns for inference to avoid leakage: {leak_cols}")
        df_input = df_input.drop(columns=leak_cols)

    df_input = df_input.copy()
    df_input["_original_index"] = np.arange(len(df_input))

    print("Running self-consistent inference...")
    predictions, dataset = predict_self_consistent(
        model,
        df_input,
        y_mean,
        y_std,
        **input_stats,
        max_iters=args.max_iters,
        tol=args.sc_tol,
        batch_size=args.batch_size,
        verbose=True,
    )

    aton_pred = predictions
    pred_mean = aton_pred
    if "residual_adapter" in checkpoint:
        print("Applying ATOM residual adapter...")
        pred_mean, _ = apply_aton_residual_adapter(
            checkpoint["residual_adapter"],
            aton_pred,
            dataset,
        )

    pred_std = None
    if args.mc_samples > 0:
        print(f"Running final MC dropout pass: {args.mc_samples} samples")
        aton_mc_mean, pred_std = predict_mc(
            model,
            dataset,
            y_mean,
            y_std,
            n_samples=args.mc_samples,
            batch_size=args.batch_size,
        )
        if "residual_adapter" in checkpoint:
            pred_mean, _ = apply_aton_residual_adapter(
                checkpoint["residual_adapter"],
                aton_mc_mean,
                dataset,
            )
        else:
            pred_mean = aton_mc_mean

    output = dataset.df.copy()
    output["ATOM_Predicted_Strength_MPa"] = pred_mean[:, 0]
    output["ATOM_Predicted_Plasticity_%"] = pred_mean[:, 1]
    if pred_std is not None:
        output["Uncertainty_Strength_MPa"] = pred_std[:, 0]
        output["Uncertainty_Plasticity_%"] = pred_std[:, 1]

    output = output.sort_values("_sc_index").reset_index(drop=True)
    drop_cols = ["_sc_index", "_original_index"] + PRED_TARGET_COLS
    output = output.drop(columns=[col for col in drop_cols if col in output.columns])

    write_output(output, args.output)
    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
