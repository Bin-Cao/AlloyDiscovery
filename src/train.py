from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from .data import (
    COMPOSITION_COLS,
    PRED_TARGET_COLS,
    PROCESS_CAT_COLS,
    PROCESS_CONT_COLS,
    TARGET_COLS,
    TEST_COLS,
    TabularDataset,
)


def _predict_standardized(model, dataset, batch_size=64):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            preds.append(model(batch))
    return torch.cat(preds, dim=0)


def denormalize(pred, y_mean, y_std):
    if isinstance(pred, torch.Tensor):
        y_mean_t = torch.as_tensor(y_mean, dtype=pred.dtype, device=pred.device)
        y_std_t = torch.as_tensor(y_std, dtype=pred.dtype, device=pred.device)
        return pred * y_std_t + y_mean_t
    return pred * y_std + y_mean


def predict_self_consistent(
    model,
    df,
    y_mean,
    y_std,
    proc_cont_mean=None,
    proc_cont_std=None,
    test_mean=None,
    test_std=None,
    state_mean=None,
    state_std=None,
    max_iters=30,
    tol=1e-3,
    batch_size=64,
    verbose=False,
):
    """
    Label-free self-consistent prediction.

    Initial t is sorted by input row order inside each composition. Later
    iterations sort t by predicted sigma_i * epsilon_i inside each composition.
    """
    if max_iters < 1:
        raise ValueError("max_iters must be >= 1")

    df_work = df.copy()
    df_work["_sc_index"] = np.arange(len(df_work))
    if "_original_index" not in df_work.columns:
        df_work["_original_index"] = np.arange(len(df_work))

    label_cols_present = [col for col in TARGET_COLS if col in df_work.columns]
    feature_df = df_work.drop(columns=label_cols_present)

    prev_pred_orig = None
    predictions = None
    dataset = None

    for iteration in range(1, max_iters + 1):
        source = "initial" if iteration == 1 else "predictions"
        dataset = TabularDataset(
            feature_df,
            trajectory_source=source,
            y_mean=y_mean,
            y_std=y_std,
            proc_cont_mean=proc_cont_mean,
            proc_cont_std=proc_cont_std,
            test_mean=test_mean,
            test_std=test_std,
            state_mean=state_mean,
            state_std=state_std,
            standardize_targets=True,
            require_targets=False,
        )
        pred_std = _predict_standardized(model, dataset, batch_size=batch_size)
        pred = denormalize(pred_std, y_mean, y_std).cpu().numpy()

        orig_idx = dataset.df["_sc_index"].values.astype(int)
        pred_orig = np.zeros_like(pred)
        pred_orig[orig_idx] = pred

        predictions = pred
        if prev_pred_orig is not None:
            rel_change = float(
                (np.abs(pred_orig - prev_pred_orig) / (np.abs(prev_pred_orig) + 1e-8)).mean()
            )
            if verbose:
                print(f"  SC iter {iteration}: mean relative change={rel_change:.6f}")
            if rel_change < tol:
                break
        elif verbose:
            print("  SC iter 1: initialized t by row order")

        feature_df[PRED_TARGET_COLS[0]] = pred_orig[:, 0]
        feature_df[PRED_TARGET_COLS[1]] = pred_orig[:, 1]
        prev_pred_orig = pred_orig

    return predictions, dataset


def make_checkpoint_payload(
    model,
    optimizer,
    epoch,
    train_loss,
    val_loss=None,
    extra=None,
):
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "columns": {
            "composition": COMPOSITION_COLS,
            "process_cat": PROCESS_CAT_COLS,
            "process_cont": PROCESS_CONT_COLS,
            "test": TEST_COLS,
            "targets": TARGET_COLS,
        },
    }
    if val_loss is not None:
        payload["val_loss"] = val_loss
    if extra:
        payload.update(extra)
    return payload


def train_model(
    model,
    train_dataset,
    val_df=None,
    epochs=100,
    batch_size=32,
    lr=1e-3,
    save_path="checkpoints/model_best.pth",
    save_every=0,
    early_stopping_patience=20,
    weight_decay=1e-4,
    task_weights=None,
    sc_max_iters=30,
    sc_tol=1e-3,
    checkpoint_extra=None,
):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    opt = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=10
    )

    if task_weights is None:
        task_weights_t = torch.tensor([1.0, 1.0])
    else:
        task_weights_t = torch.tensor(task_weights, dtype=torch.float32)
    loss_fn = torch.nn.MSELoss(reduction="none")

    best_loss = float("inf")
    patience_counter = 0
    best_epoch = -1

    for ep in range(epochs):
        model.train()
        total_train = 0.0
        for batch in train_loader:
            pred = model(batch)
            loss = (loss_fn(pred, batch["y"]) * task_weights_t).mean()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            total_train += loss.item()

        avg_train_loss = total_train / max(len(train_loader), 1)

        avg_val_loss = None
        score_loss = avg_train_loss
        if val_df is not None:
            pred_denorm, _ = predict_self_consistent(
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
                max_iters=sc_max_iters,
                tol=sc_tol,
                batch_size=batch_size * 2,
                verbose=False,
            )
            y_true = val_df[TARGET_COLS].values.astype(np.float32)
            y_pred_std = (pred_denorm - train_dataset.y_mean) / train_dataset.y_std
            y_true_std = (y_true - train_dataset.y_mean) / train_dataset.y_std
            avg_val_loss = float(((y_pred_std - y_true_std) ** 2 * task_weights_t.numpy()).mean())
            score_loss = avg_val_loss
            scheduler.step(avg_val_loss)
            print(
                f"Epoch {ep}: train_loss={avg_train_loss:.4f}, "
                f"self_consistent_val_loss={avg_val_loss:.4f}"
            )
        else:
            scheduler.step(avg_train_loss)
            print(f"Epoch {ep}: train_loss={avg_train_loss:.4f}")

        if score_loss < best_loss:
            best_loss = score_loss
            best_epoch = ep
            patience_counter = 0
            payload = make_checkpoint_payload(
                model,
                opt,
                ep,
                avg_train_loss,
                val_loss=avg_val_loss,
                extra=checkpoint_extra,
            )
            torch.save(payload, save_path)
            print(f"  Saved best model to {save_path} (loss: {best_loss:.4f})")
        else:
            patience_counter += 1

        if save_every and (ep + 1) % save_every == 0:
            periodic_path = save_path.with_name(f"{save_path.stem}_epoch{ep + 1}{save_path.suffix}")
            torch.save(
                make_checkpoint_payload(
                    model,
                    opt,
                    ep,
                    avg_train_loss,
                    val_loss=avg_val_loss,
                    extra=checkpoint_extra,
                ),
                periodic_path,
            )

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {ep + 1}; best epoch={best_epoch}")
            break

    return best_loss, best_epoch


def compute_metrics(y_true, y_pred):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    metrics = {}
    for i, name in enumerate(["Strength_MPa", "Plasticity_%"]):
        y_t = y_true[:, i]
        y_p = y_pred[:, i]
        mse = float(((y_t - y_p) ** 2).mean())
        rmse = float(np.sqrt(mse))
        mae = float(np.abs(y_t - y_p).mean())
        ss_res = float(((y_t - y_p) ** 2).sum())
        ss_tot = float(((y_t - y_t.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        metrics[name] = {"RMSE": rmse, "MAE": mae, "R2": r2}
    return metrics


def print_metrics(metrics):
    print("\nEvaluation Metrics")
    print("=" * 70)
    for target_name, vals in metrics.items():
        print(f"{target_name}: RMSE={vals['RMSE']:.4f}, MAE={vals['MAE']:.4f}, R2={vals['R2']:.4f}")
