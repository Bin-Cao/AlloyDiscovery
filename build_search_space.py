"""Generate the feature-only search space for the current data.xlsx schema.

Only Cold CR_%, annealing temperature, and annealing time are scanned.
Rate_S-1 is fixed at 1e-3. The remaining feature values are copied from one
reference row, preserving the original search-space logic.
"""

import argparse

import numpy as np
import pandas as pd

from src.data import TARGET_COLS, read_alloy_data, validate_feature_columns


CONFIG = {
    "data_path": "data.xlsx",
    "sheet_name": 0,
    "output": "search_space.csv",
    "reference_row": 0,
    "cr_step": 1.0,
    "annealing_temp_step": 5.0,
    "annealing_time_step": 0.5,
    "rate_fixed": 1e-3,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Build aligned alloy search space.")
    parser.add_argument("--data", default=CONFIG["data_path"])
    parser.add_argument("--sheet", default=CONFIG["sheet_name"])
    parser.add_argument("--output", default=CONFIG["output"])
    parser.add_argument("--reference-row", type=int, default=CONFIG["reference_row"])
    return parser.parse_args()


def normalize_sheet_name(sheet):
    if isinstance(sheet, str) and sheet.isdigit():
        return int(sheet)
    return sheet


def main() -> None:
    args = parse_args()
    df = read_alloy_data(args.data, sheet_name=normalize_sheet_name(args.sheet))
    validate_feature_columns(df, require_targets=True)

    feature_cols = [col for col in df.columns if col not in TARGET_COLS]
    if args.reference_row < 0 or args.reference_row >= len(df):
        raise ValueError(f"reference-row must be between 0 and {len(df) - 1}")
    sample = df.iloc[args.reference_row]

    cr_grid = np.arange(
        df["Cold CR_%"].min(),
        df["Cold CR_%"].max() + CONFIG["cr_step"] / 2,
        CONFIG["cr_step"],
    )
    temp_grid = np.arange(
        df["Annealing_Temp_K"].min(),
        df["Annealing_Temp_K"].max() + CONFIG["annealing_temp_step"] / 2,
        CONFIG["annealing_temp_step"],
    )
    time_grid = np.arange(
        df["Annealing_time_h"].min(),
        df["Annealing_time_h"].max() + CONFIG["annealing_time_step"] / 2,
        CONFIG["annealing_time_step"],
    )

    print(f"Cold CR_% grid:          {len(cr_grid)} values [{cr_grid[0]} .. {cr_grid[-1]}]")
    print(f"Annealing_Temp_K grid:   {len(temp_grid)} values [{temp_grid[0]} .. {temp_grid[-1]}]")
    print(f"Annealing_time_h grid:   {len(time_grid)} values [{time_grid[0]} .. {time_grid[-1]}]")
    print(f"Total combinations:      {len(cr_grid) * len(temp_grid) * len(time_grid):,}")

    cr_mesh, temp_mesh, time_mesh = np.meshgrid(
        cr_grid, temp_grid, time_grid, indexing="ij"
    )
    n = cr_mesh.size

    out = pd.DataFrame({col: np.full(n, sample[col]) for col in feature_cols})
    out["Cold CR_%"] = cr_mesh.ravel()
    out["Annealing_Temp_K"] = temp_mesh.ravel()
    out["Annealing_time_h"] = time_mesh.ravel()
    out["Rate_S-1"] = CONFIG["rate_fixed"]
    out = out[feature_cols]

    if args.output.lower().endswith(".xlsx"):
        out.to_excel(args.output, index=False)
    else:
        out.to_csv(args.output, index=False)
    print(f"Wrote {len(out):,} rows to {args.output}")


if __name__ == "__main__":
    main()
