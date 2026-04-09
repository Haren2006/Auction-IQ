#!/usr/bin/env python3
"""Inference adapter for the Auction IQ Streamlit app."""

from __future__ import annotations

import json
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.week9.predict_week9 import _factor_lists, load_week9_artifacts


APP_ROOT = Path(__file__).resolve().parent
MODELS_DIR = APP_ROOT / "models"

ITEM_OPTIONS = (
    "Cartier wristwatch",
    "Palm Pilot M515 PDA",
    "Xbox game console",
)
AUCTION_TYPES = (
    "3 day auction",
    "5 day auction",
    "7 day auction",
)
CHECKPOINTS = (25, 50, 75, 85, 90, 95, 100)
REQUIRED_ARTIFACTS = (
    "point_model.pkl",
    "metadata.json",
    "week9_metrics.json",
    "quantile_models.pkl",
    "week11_quantile_metadata.json",
    "week11_quantile_metrics.json",
)
DEFAULT_LEADING_BIDDER_RATE = 6.0


def required_artifact_paths() -> List[Path]:
    return [MODELS_DIR / name for name in REQUIRED_ARTIFACTS]


def missing_artifact_paths() -> List[Path]:
    return [path for path in required_artifact_paths() if not path.exists()]


def _dedupe_strings(values: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


@lru_cache(maxsize=1)
def load_point_bundle() -> tuple[Dict[str, Any], Dict[str, Any]]:
    return load_week9_artifacts(MODELS_DIR)


@lru_cache(maxsize=1)
def load_quantile_bundle() -> tuple[Dict[str, Any], Dict[str, Any]]:
    artifact_path = MODELS_DIR / "quantile_models.pkl"
    metadata_path = MODELS_DIR / "week11_quantile_metadata.json"
    if not artifact_path.exists():
        raise FileNotFoundError(f"Missing model artifact: {artifact_path.resolve()}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata artifact: {metadata_path.resolve()}")

    with artifact_path.open("rb") as handle:
        model_artifact = pickle.load(handle)
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return model_artifact, metadata


@lru_cache(maxsize=1)
def load_week9_metrics() -> Dict[str, Any]:
    path = MODELS_DIR / "week9_metrics.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def load_week11_metrics() -> Dict[str, Any]:
    path = MODELS_DIR / "week11_quantile_metrics.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def point_metadata() -> Dict[str, Any]:
    _, metadata = load_point_bundle()
    return metadata


def quantile_metadata() -> Dict[str, Any]:
    _, metadata = load_quantile_bundle()
    return metadata


def auction_days_from_type(auction_type: str) -> float:
    mapping = {
        "3 day auction": 3.0,
        "5 day auction": 5.0,
        "7 day auction": 7.0,
    }
    if auction_type not in mapping:
        raise ValueError(f"Unsupported auction_type: {auction_type}")
    return mapping[auction_type]


def snapshot_pct_string(progress: float) -> str:
    return f"{float(progress):.2f}"


def _leading_bidder_rate_default() -> float:
    medians = point_metadata().get("numeric_feature_medians", {})
    value = medians.get("leading_bidder_rate_so_far", DEFAULT_LEADING_BIDDER_RATE)
    try:
        return float(value)
    except (TypeError, ValueError):
        return DEFAULT_LEADING_BIDDER_RATE


def validate_snapshot(snapshot: Dict[str, Any]) -> List[str]:
    issues: List[str] = []
    if snapshot["current_price"] < snapshot["opening_bid"]:
        issues.append("Current price is below opening bid. Check the input values.")
    if snapshot["highest_observed_bid"] < snapshot["current_price"]:
        issues.append("Highest observed bid should usually be at least the current price.")
    if snapshot["num_unique_bidders_so_far"] > snapshot["num_bids_so_far"]:
        issues.append("Unique bidders cannot exceed total bids.")
    if snapshot["auction_type"] not in AUCTION_TYPES:
        issues.append("Auction type must be one of the supported trained categories.")
    return issues


def map_ui_snapshot_to_model(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    auction_type = str(snapshot["auction_type"])
    auction_days = auction_days_from_type(auction_type)
    progress = float(snapshot["auction_progress"])
    current_price = float(snapshot["current_price"])
    highest_observed_bid = max(float(snapshot["highest_observed_bid"]), current_price)

    return {
        "item_name": str(snapshot["item_name"]),
        "auction_type": auction_type,
        "snapshot_pct": snapshot_pct_string(progress),
        "auction_days": auction_days,
        "snapshot_time_days": round(auction_days * progress, 2),
        "opening_bid": float(snapshot["opening_bid"]),
        "bid_count_so_far": int(snapshot["num_bids_so_far"]),
        "unique_bidders_so_far": int(snapshot["num_unique_bidders_so_far"]),
        "max_observed_bid_so_far": highest_observed_bid,
        "leading_bidder_rate_so_far": _leading_bidder_rate_default(),
    }


def predict_point(snapshot_record: Dict[str, Any], current_price: float | None = None) -> Dict[str, Any]:
    model_artifact, metadata = load_point_bundle()
    feature_columns = model_artifact["feature_columns"]
    pipeline = model_artifact["pipeline"]

    row = {feature: snapshot_record.get(feature) for feature in feature_columns}
    frame = pd.DataFrame([row])
    point_estimate = float(pipeline.predict(frame)[0])
    if current_price is not None:
        point_estimate = max(point_estimate, float(current_price))

    top_positive, top_negative = _factor_lists(snapshot_record=row, metadata=metadata, top_k=3)
    limitations = _dedupe_strings(
        metadata.get("limitations", []) + quantile_metadata().get("limitations", [])
    )

    return {
        "point_estimate": round(point_estimate, 2),
        "top_positive_factors": top_positive,
        "top_negative_factors": top_negative,
        "limitations": limitations,
    }


def predict_quantiles(snapshot_record: Dict[str, Any], current_price: float) -> Dict[str, float]:
    model_artifact, metadata = load_quantile_bundle()
    feature_columns = model_artifact["feature_columns"]
    models = model_artifact["models"]
    ordered_labels = model_artifact.get("postprocess", {}).get(
        "ordered_labels",
        metadata.get("postprocess", {}).get("ordered_labels", ["q10", "q50", "q75", "q90"]),
    )

    row = {feature: snapshot_record.get(feature) for feature in feature_columns}
    frame = pd.DataFrame([row])
    raw = np.array([float(models[label].predict(frame)[0]) for label in ordered_labels], dtype=float)
    ordered = np.maximum.accumulate(raw)
    ordered = np.maximum(ordered, float(current_price))
    ordered = np.maximum.accumulate(ordered)

    return {
        label: round(float(value), 2)
        for label, value in zip(ordered_labels, ordered)
    }


def predict_snapshot(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    model_input = map_ui_snapshot_to_model(snapshot)
    point_result = predict_point(model_input, current_price=float(snapshot["current_price"]))
    quantiles = predict_quantiles(model_input, current_price=float(snapshot["current_price"]))

    return {
        "model_input": model_input,
        "point_estimate": point_result["point_estimate"],
        "quantiles": quantiles,
        "top_positive_factors": point_result["top_positive_factors"],
        "top_negative_factors": point_result["top_negative_factors"],
        "limitations": point_result["limitations"],
    }


def build_explanation(
    snapshot: Dict[str, Any],
    prediction: Dict[str, Any],
    mode: str,
) -> Dict[str, str]:
    progress_pct = int(snapshot["auction_progress"] * 100)
    point_estimate = prediction["point_estimate"]
    quantiles = prediction["quantiles"]
    positives = " ".join(prediction["top_positive_factors"][:2])
    cautions = " ".join(prediction["top_negative_factors"][:1])

    what_model_expects = (
        f"The backend uses the Week 9 point model and Week 11 quantile models on a "
        f"{snapshot['auction_type']} snapshot at {progress_pct}% progress. "
        f"This input includes {snapshot['num_bids_so_far']} bids so far, "
        f"{snapshot['num_unique_bidders_so_far']} unique bidders, and "
        f"a highest observed bid of ${snapshot['highest_observed_bid']:,.2f}."
    )

    why = (
        f"The point estimate is ${point_estimate:,.2f} with a q10 to q90 range of "
        f"${quantiles['q10']:,.2f} to ${quantiles['q90']:,.2f}. "
        f"{positives} {cautions}".strip()
    )

    if mode == "buyer":
        suggested_action = (
            "Use q50 for conservative bidding, q75 for balanced bidding, and q90 for "
            "aggressive bidding. If the live price is already above your selected "
            "threshold, the recommendation becomes PASS."
        )
    else:
        suggested_action = (
            "Use the seller scenario sweep to compare opening-bid changes with the real "
            "backend predictions. Treat the chart as directional guidance rather than a guarantee."
        )

    limitations = " ".join(prediction["limitations"])

    return {
        "What the model expects": what_model_expects,
        "Why this prediction was generated": why,
        "Suggested action": suggested_action,
        "Model limitations": limitations,
    }


def seller_scenarios(snapshot: Dict[str, Any]) -> pd.DataFrame:
    base_prediction = predict_snapshot(snapshot)
    base_point = base_prediction["point_estimate"]
    rows = []

    for pct in (-0.20, -0.10, 0.0, 0.10, 0.20):
        modified = dict(snapshot)
        modified["opening_bid"] = round(max(0.01, snapshot["opening_bid"] * (1 + pct)), 2)
        prediction = predict_snapshot(modified)
        rows.append(
            {
                "Scenario": f"{pct:+.0%}",
                "Opening bid": round(modified["opening_bid"], 2),
                "Predicted final price": prediction["point_estimate"],
                "Change vs current setup": round(prediction["point_estimate"] - base_point, 2),
            }
        )

    return pd.DataFrame(rows)
