#!/usr/bin/env python3
"""Week 9 minimal prediction interface for point estimates."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


DEFAULT_LIMITATIONS = [
    "This estimate is based on historical sold-auction snapshots only.",
    "The model does not use seller reputation, item condition text, images, or calendar timing.",
    "Proxy-bidding behavior can make bid progression noisy.",
]


def load_week9_artifacts(models_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    point_model_path = models_dir / "point_model.pkl"
    metadata_path = models_dir / "metadata.json"
    if not point_model_path.exists():
        raise FileNotFoundError(f"Missing model artifact: {point_model_path.resolve()}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata artifact: {metadata_path.resolve()}")

    with point_model_path.open("rb") as f:
        model_artifact = pickle.load(f)
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    return model_artifact, metadata


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _factor_lists(
    snapshot_record: Dict[str, Any], metadata: Dict[str, Any], top_k: int = 3
) -> Tuple[List[str], List[str]]:
    importance = metadata.get("feature_importance_raw", {})
    medians = metadata.get("numeric_feature_medians", {})

    ranked_features = [k for k, _ in sorted(importance.items(), key=lambda kv: kv[1], reverse=True)]
    positive: List[str] = []
    negative: List[str] = []

    for feat in ranked_features:
        if len(positive) >= top_k and len(negative) >= top_k:
            break
        val = snapshot_record.get(feat)
        med = medians.get(feat)
        valf = _to_float(val)
        medf = _to_float(med)
        if valf is None or medf is None:
            continue

        if valf >= medf and len(positive) < top_k:
            positive.append(f"{feat} is above typical level ({valf:.2f} vs {medf:.2f}).")
        elif valf < medf and len(negative) < top_k:
            negative.append(f"{feat} is below typical level ({valf:.2f} vs {medf:.2f}).")

    if len(positive) < top_k:
        for feat in ranked_features:
            if len(positive) >= top_k:
                break
            if snapshot_record.get(feat) not in (None, "", "MISSING"):
                msg = f"{feat} contributes meaningful signal in this estimate."
                if msg not in positive:
                    positive.append(msg)

    if len(negative) < top_k:
        fill_msg = "No dominant downside feature exceeded the top-factor threshold."
        while len(negative) < top_k:
            negative.append(fill_msg)

    return positive[:top_k], negative[:top_k]


def predict_point(snapshot_record: Dict[str, Any], models_dir: str | Path = "models") -> Dict[str, Any]:
    models_path = Path(models_dir)
    model_artifact, metadata = load_week9_artifacts(models_path)

    feature_columns = model_artifact["feature_columns"]
    pipeline = model_artifact["pipeline"]

    row = {feature: snapshot_record.get(feature) for feature in feature_columns}
    frame = pd.DataFrame([row])
    point_estimate = float(pipeline.predict(frame)[0])

    top_positive, top_negative = _factor_lists(snapshot_record=row, metadata=metadata, top_k=3)

    return {
        "point_estimate": round(point_estimate, 4),
        "top_positive_factors": top_positive,
        "top_negative_factors": top_negative,
        "limitations": metadata.get("limitations", DEFAULT_LIMITATIONS),
    }


def _default_sample(metadata: Dict[str, Any]) -> Dict[str, Any]:
    medians = metadata.get("numeric_feature_medians", {})
    return {
        "item_name": "Palm Pilot M515 PDA",
        "auction_type": "7 day auction",
        "snapshot_pct": "0.50",
        "auction_days": medians.get("auction_days", 7.0),
        "snapshot_time_days": medians.get("snapshot_time_days", 3.5),
        "opening_bid": medians.get("opening_bid", 50.0),
        "bid_count_so_far": medians.get("bid_count_so_far", 4.0),
        "unique_bidders_so_far": medians.get("unique_bidders_so_far", 3.0),
        "max_observed_bid_so_far": medians.get("max_observed_bid_so_far", 100.0),
        "leading_bidder_rate_so_far": medians.get("leading_bidder_rate_so_far", 10.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Week 9 point prediction.")
    parser.add_argument("--models-dir", default=Path("models"), type=Path)
    parser.add_argument(
        "--sample-json",
        default=None,
        help="Optional JSON string for a snapshot-like input record.",
    )
    args = parser.parse_args()

    _, metadata = load_week9_artifacts(args.models_dir)
    if args.sample_json:
        sample = json.loads(args.sample_json)
    else:
        sample = _default_sample(metadata)

    output = predict_point(snapshot_record=sample, models_dir=args.models_dir)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
