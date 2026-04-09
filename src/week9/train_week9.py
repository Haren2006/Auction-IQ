#!/usr/bin/env python3
"""Week 9 training pipeline for snapshot-based point price prediction.

This script intentionally consumes teammate-produced snapshot data as-is.
It trains baselines + a RandomForest point model and writes Week 9 artifacts.
"""

from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TARGET_COLUMN = "final_price"
GROUP_COLUMN = "auction_id"

REQUIRED_COLUMNS = [
    "auction_id",
    "item_name",
    "auction_type",
    "auction_days",
    "snapshot_pct",
    "snapshot_time_days",
    "estimated_bid_amount",
    "opening_bid",
    "final_price",
    "bid_count_so_far",
    "unique_bidders_so_far",
    "max_observed_bid_so_far",
    "leading_bidder_so_far",
    "leading_bidder_rate_so_far",
]

CATEGORICAL_COLUMNS = ["item_name", "auction_type", "snapshot_pct"]
NUMERIC_COLUMNS = [
    "auction_days",
    "snapshot_time_days",
    "opening_bid",
    "bid_count_so_far",
    "unique_bidders_so_far",
    "max_observed_bid_so_far",
    "leading_bidder_rate_so_far",
]
FEATURE_COLUMNS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Week 9 point model artifacts.")
    parser.add_argument(
        "--input-csv",
        default="auction_snapshots.csv",
        type=Path,
        help="Path to teammate snapshot CSV.",
    )
    parser.add_argument(
        "--models-dir",
        default=Path("models"),
        type=Path,
        help="Directory for output artifacts.",
    )
    parser.add_argument("--test-size", default=0.2, type=float, help="Test set proportion.")
    parser.add_argument("--random-state", default=42, type=int, help="Random seed.")
    parser.add_argument(
        "--n-estimators", default=300, type=int, help="RandomForest n_estimators."
    )
    return parser.parse_args()


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {"mae": mae, "rmse": rmse}


def grouped_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, groups: Iterable
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    grp = pd.Series(groups).astype(str)
    for value in sorted(grp.unique(), key=str):
        idx = grp == value
        if int(idx.sum()) == 0:
            continue
        out[str(value)] = metric_dict(y_true[idx], y_pred[idx])
    return out


def load_and_validate(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing input CSV: {csv_path.resolve()}")

    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    numeric_to_parse = [
        "auction_days",
        "snapshot_pct",
        "snapshot_time_days",
        "estimated_bid_amount",
        "opening_bid",
        "final_price",
        "bid_count_so_far",
        "unique_bidders_so_far",
        "max_observed_bid_so_far",
        "leading_bidder_rate_so_far",
    ]
    for col in numeric_to_parse:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep snapshot stage categorical for modeling/reporting.
    df["snapshot_pct"] = df["snapshot_pct"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "MISSING")
    for col in ["item_name", "auction_type", "leading_bidder_so_far"]:
        df[col] = df[col].astype("string").fillna("MISSING").astype(str)

    # Group id as stable string.
    df[GROUP_COLUMN] = df[GROUP_COLUMN].astype("string").fillna("MISSING").astype(str)

    before = len(df)
    df = df.dropna(subset=[TARGET_COLUMN]).copy()
    dropped_target = before - len(df)
    if dropped_target:
        print(f"Dropped {dropped_target} rows with missing target ({TARGET_COLUMN}).")

    return df


def make_split(
    df: pd.DataFrame, test_size: float, random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, df[TARGET_COLUMN], groups=df[GROUP_COLUMN]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    overlap = set(train_df[GROUP_COLUMN]).intersection(set(test_df[GROUP_COLUMN]))
    if overlap:
        raise RuntimeError(f"Leakage guard failed: {len(overlap)} auction IDs overlap train/test.")

    return train_df, test_df


def baseline_item_median(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    global_median = float(train_df[TARGET_COLUMN].median())
    by_item = train_df.groupby("item_name")[TARGET_COLUMN].median().to_dict()
    return test_df["item_name"].map(by_item).fillna(global_median).to_numpy(dtype=float)


def baseline_opening_linear(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    med_open = float(train_df["opening_bid"].median())
    x_train = train_df[["opening_bid"]].fillna(med_open).to_numpy(dtype=float)
    x_test = test_df[["opening_bid"]].fillna(med_open).to_numpy(dtype=float)
    y_train = train_df[TARGET_COLUMN].to_numpy(dtype=float)
    model = LinearRegression()
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    return np.clip(pred, a_min=0.0, a_max=None)


def baseline_current_price(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    global_median = float(train_df[TARGET_COLUMN].median())
    pred = test_df["max_observed_bid_so_far"].where(
        test_df["max_observed_bid_so_far"].notna(), test_df["opening_bid"]
    )
    pred = pred.fillna(global_median).to_numpy(dtype=float)
    return np.clip(pred, a_min=0.0, a_max=None)


def train_point_model(train_df: pd.DataFrame, random_state: int, n_estimators: int) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_COLUMNS,
            ),
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), NUMERIC_COLUMNS),
        ],
        remainder="drop",
    )

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        min_samples_leaf=1,
    )
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", rf)])

    x_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN].to_numpy(dtype=float)
    pipeline.fit(x_train, y_train)
    return pipeline


def aggregate_raw_feature_importance(model_pipeline: Pipeline) -> Dict[str, float]:
    pre = model_pipeline.named_steps["preprocessor"]
    model = model_pipeline.named_steps["model"]

    names = pre.get_feature_names_out()
    importances = model.feature_importances_
    agg: Dict[str, float] = {name: 0.0 for name in FEATURE_COLUMNS}

    for encoded_name, imp in zip(names, importances):
        if encoded_name.startswith("num__"):
            raw = encoded_name.replace("num__", "", 1)
        elif encoded_name.startswith("cat__"):
            tail = encoded_name.replace("cat__", "", 1)
            raw = None
            for col in CATEGORICAL_COLUMNS:
                prefix = f"{col}_"
                if tail == col or tail.startswith(prefix):
                    raw = col
                    break
            if raw is None:
                raw = tail.split("_", 1)[0]
        else:
            raw = encoded_name
        agg[raw] = float(agg.get(raw, 0.0) + float(imp))

    sorted_items = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)
    return {k: float(v) for k, v in sorted_items}


def ensure_serializable_metrics(d: Dict) -> Dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[str(k)] = ensure_serializable_metrics(v)
        elif isinstance(v, np.floating):
            out[str(k)] = float(v)
        elif isinstance(v, np.integer):
            out[str(k)] = int(v)
        else:
            out[str(k)] = v
    return out


def main() -> None:
    args = parse_args()
    models_dir = args.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_validate(args.input_csv)
    train_df, test_df = make_split(df, test_size=args.test_size, random_state=args.random_state)

    y_test = test_df[TARGET_COLUMN].to_numpy(dtype=float)

    preds_item = baseline_item_median(train_df, test_df)
    preds_open = baseline_opening_linear(train_df, test_df)
    preds_curr = baseline_current_price(train_df, test_df)

    point_model = train_point_model(
        train_df=train_df, random_state=args.random_state, n_estimators=args.n_estimators
    )
    preds_rf = point_model.predict(test_df[FEATURE_COLUMNS])

    all_preds = {
        "item_median_baseline": preds_item,
        "opening_bid_linear_baseline": preds_open,
        "current_price_heuristic_baseline": preds_curr,
        "random_forest_point_model": preds_rf,
    }

    metrics = {
        "dataset": {
            "input_csv": str(args.input_csv.resolve()),
            "rows_total": int(len(df)),
            "rows_train": int(len(train_df)),
            "rows_test": int(len(test_df)),
            "auctions_total": int(df[GROUP_COLUMN].nunique()),
            "auctions_train": int(train_df[GROUP_COLUMN].nunique()),
            "auctions_test": int(test_df[GROUP_COLUMN].nunique()),
            "snapshot_pct_values": sorted([str(v) for v in df["snapshot_pct"].dropna().unique()]),
        },
        "leakage_check": {
            "train_test_auction_overlap_count": int(
                len(set(train_df[GROUP_COLUMN]).intersection(set(test_df[GROUP_COLUMN])))
            )
        },
        "overall": {},
        "by_snapshot_pct": {},
        "by_item_name": {},
    }

    for model_name, pred in all_preds.items():
        metrics["overall"][model_name] = metric_dict(y_test, pred)
        metrics["by_snapshot_pct"][model_name] = grouped_metrics(
            y_test, pred, test_df["snapshot_pct"].to_numpy()
        )
        metrics["by_item_name"][model_name] = grouped_metrics(
            y_test, pred, test_df["item_name"].to_numpy()
        )

    feature_importance_raw = aggregate_raw_feature_importance(point_model)
    numeric_medians = (
        train_df[NUMERIC_COLUMNS].median().replace({np.nan: None}).to_dict()  # type: ignore[arg-type]
    )
    numeric_medians = {
        k: (float(v) if v is not None and pd.notna(v) else None) for k, v in numeric_medians.items()
    }

    model_artifact = {
        "pipeline": point_model,
        "feature_columns": FEATURE_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "numeric_columns": NUMERIC_COLUMNS,
        "target_column": TARGET_COLUMN,
    }
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(args.input_csv.resolve()),
        "feature_columns": FEATURE_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "numeric_columns": NUMERIC_COLUMNS,
        "target_column": TARGET_COLUMN,
        "group_column": GROUP_COLUMN,
        "snapshot_stage_column": "snapshot_pct",
        "split": {
            "random_state": int(args.random_state),
            "test_size": float(args.test_size),
            "rows_train": int(len(train_df)),
            "rows_test": int(len(test_df)),
            "auctions_train": int(train_df[GROUP_COLUMN].nunique()),
            "auctions_test": int(test_df[GROUP_COLUMN].nunique()),
        },
        "numeric_feature_medians": numeric_medians,
        "feature_importance_raw": feature_importance_raw,
        "limitations": [
            "Built from historical sold-auction snapshots only.",
            "No seller reputation, item condition text, images, or calendar-time context.",
            "Proxy-bidding behavior may make bidding dynamics appear non-monotonic.",
        ],
    }

    point_model_path = models_dir / "point_model.pkl"
    metrics_path = models_dir / "week9_metrics.json"
    metadata_path = models_dir / "metadata.json"

    with point_model_path.open("wb") as f:
        pickle.dump(model_artifact, f)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(ensure_serializable_metrics(metrics), f, indent=2)
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(ensure_serializable_metrics(metadata), f, indent=2)

    print("Week 9 artifacts saved:")
    print(f"  - {point_model_path.resolve()}")
    print(f"  - {metrics_path.resolve()}")
    print(f"  - {metadata_path.resolve()}")
    print("\nOverall test metrics:")
    for name, m in metrics["overall"].items():
        print(f"  {name:35s} MAE={m['mae']:.3f}  RMSE={m['rmse']:.3f}")


if __name__ == "__main__":
    main()
