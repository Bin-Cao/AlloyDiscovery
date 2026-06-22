import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Optional


COMPOSITION_COLS = [
    "Al", "Co", "Cr", "Fe", "Ni", "Ti", "Mo", "Nb",
]

PROCESS_CAT_COLS = [
    "Eutectic", "Preparation", "Processing", "Tensile/compress",
]

PROCESS_CONT_COLS = [
    "Cold CR_%",
    "Annealing_Temp_K",
    "Annealing_time_h",
    "Aging_Temp_K",
    "Aging_Time_h",
]

TEST_COLS = [
    "Rate_S-1", "Tes_Temp_K",
]

TARGET_COLS = [
    "Strength_MPa", "Plasticity_%",
]

PRED_TARGET_COLS = [
    "_pred_strength", "_pred_plasticity",
]


def read_alloy_data(path: str, sheet_name=0) -> pd.DataFrame:
    """Read an Excel/CSV alloy table."""
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_excel(path, sheet_name=sheet_name)


def validate_feature_columns(df: pd.DataFrame, require_targets: bool = False) -> None:
    required = COMPOSITION_COLS + PROCESS_CAT_COLS + PROCESS_CONT_COLS + TEST_COLS
    if require_targets:
        required += TARGET_COLS
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def get_proc_cat_dims(df: pd.DataFrame) -> list[int]:
    validate_feature_columns(df, require_targets=False)
    return [int(df[col].max()) + 1 for col in PROCESS_CAT_COLS]


class TabularDataset(Dataset):
    """
    One row = one alloy experiment/candidate.

    Trajectory state is a single scalar t. For supervised training, t is sorted
    by sigma_i * epsilon_i within each composition. For label-free validation
    or inference, t can be initialized by row order and then updated from
    self-consistent predictions.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        trajectory_source: str = "labels",
        y_mean=None,
        y_std=None,
        proc_cont_mean=None,
        proc_cont_std=None,
        test_mean=None,
        test_std=None,
        state_mean=None,
        state_std=None,
        standardize_targets: bool = True,
        require_targets: Optional[bool] = None,
    ):
        df = df.copy()
        if require_targets is None:
            require_targets = trajectory_source == "labels"
        validate_feature_columns(df, require_targets=require_targets)

        if "_original_index" not in df.columns:
            df["_original_index"] = np.arange(len(df))

        df = self._build_trajectories(df, trajectory_source)
        self.df = df.reset_index(drop=True)

        comp = self.df[COMPOSITION_COLS].values.astype(np.float32)
        comp_sum = comp.sum(axis=1, keepdims=True) + 1e-8
        self.comp = comp / comp_sum

        self.proc_cat_codes = self.df[PROCESS_CAT_COLS].values.astype(np.int64)
        self.proc_cont_raw = self.df[PROCESS_CONT_COLS].values.astype(np.float32)
        self.test_raw = self.df[TEST_COLS].values.astype(np.float32)
        self.t = self.df["t"].values.astype(np.float32)
        self.state_raw = self.t.reshape(-1, 1).astype(np.float32)

        self.proc_cont_mean, self.proc_cont_std = self._resolve_stats(
            self.proc_cont_raw, proc_cont_mean, proc_cont_std
        )
        self.test_mean, self.test_std = self._resolve_stats(
            self.test_raw, test_mean, test_std
        )
        self.state_mean, self.state_std = self._resolve_stats(
            self.state_raw, state_mean, state_std
        )
        self.proc_cont = (self.proc_cont_raw - self.proc_cont_mean) / self.proc_cont_std
        self.test = (self.test_raw - self.test_mean) / self.test_std
        self.state = (self.state_raw - self.state_mean) / self.state_std

        self.standardize_targets = standardize_targets
        self.has_y = all(col in self.df.columns for col in TARGET_COLS)
        if self.has_y:
            self.y_raw = self.df[TARGET_COLS].values.astype(np.float32)
            if standardize_targets:
                if y_mean is None or y_std is None:
                    self.y_mean = self.y_raw.mean(axis=0, keepdims=True)
                    self.y_std = self.y_raw.std(axis=0, keepdims=True) + 1e-8
                else:
                    self.y_mean = np.asarray(y_mean, dtype=np.float32)
                    self.y_std = np.asarray(y_std, dtype=np.float32)
                self.y = (self.y_raw - self.y_mean) / self.y_std
            else:
                self.y = self.y_raw
                self.y_mean = None
                self.y_std = None
        else:
            self.y_raw = np.zeros((len(self.df), len(TARGET_COLS)), dtype=np.float32)
            self.y = self.y_raw
            self.y_mean = np.asarray(y_mean, dtype=np.float32) if y_mean is not None else None
            self.y_std = np.asarray(y_std, dtype=np.float32) if y_std is not None else None

    def _build_trajectories(self, df: pd.DataFrame, trajectory_source: str) -> pd.DataFrame:
        if trajectory_source not in {"labels", "predictions", "initial"}:
            raise ValueError(
                "trajectory_source must be one of: labels, predictions, initial"
            )

        if trajectory_source == "labels":
            score = df[TARGET_COLS[0]].astype(float) * df[TARGET_COLS[1]].astype(float)
        elif trajectory_source == "predictions":
            missing = [col for col in PRED_TARGET_COLS if col not in df.columns]
            if missing:
                raise ValueError(f"Missing prediction columns for trajectory update: {missing}")
            score = df[PRED_TARGET_COLS[0]].astype(float) * df[PRED_TARGET_COLS[1]].astype(float)
        else:
            score = pd.Series(np.arange(len(df), dtype=float), index=df.index)

        df["_trajectory_score"] = score.values
        df["_comp_id"] = df.groupby(COMPOSITION_COLS, sort=False).ngroup()

        trajectories = []
        for _, group in df.groupby("_comp_id", sort=False):
            group = group.sort_values(
                ["_trajectory_score", "_original_index"],
                ascending=[True, True],
                kind="mergesort",
            ).copy()
            group["t"] = np.arange(1, len(group) + 1, dtype=np.float32)
            trajectories.append(group)

        out = pd.concat(trajectories, ignore_index=True)
        return out.drop(columns=["_trajectory_score", "_comp_id"])

    @staticmethod
    def _resolve_stats(values, mean, std):
        if mean is None or std is None:
            resolved_mean = values.mean(axis=0, keepdims=True)
            resolved_std = values.std(axis=0, keepdims=True) + 1e-8
        else:
            resolved_mean = np.asarray(mean, dtype=np.float32)
            resolved_std = np.asarray(std, dtype=np.float32)
        return resolved_mean.astype(np.float32), resolved_std.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "comp": torch.tensor(self.comp[idx]),
            "proc_cat": torch.tensor(self.proc_cat_codes[idx]),
            "proc_cont": torch.tensor(self.proc_cont[idx]),
            "test": torch.tensor(self.test[idx]),
            "state": torch.tensor(self.state[idx]),
            "y": torch.tensor(self.y[idx]),
        }
