from __future__ import annotations

import os
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from auction_iq_backend import (
    AUCTION_TYPES,
    CHECKPOINTS,
    ITEM_OPTIONS,
    build_explanation,
    load_week11_metrics,
    load_week9_metrics,
    missing_artifact_paths,
    predict_snapshot,
    seller_scenarios,
    validate_snapshot,
)
from auction_iq_llm import LLM_OPTIONS, get_llm_response


st.set_page_config(
    page_title="Auction IQ",
    page_icon="AIQ",
    layout="wide",
    initial_sidebar_state="expanded",
)



st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2rem;
    }
    .auction-card {
        border: 1px solid rgba(49, 51, 63, 0.15);
        border-radius: 16px;
        padding: 1rem 1rem 0.8rem 1rem;
        background: rgba(250, 250, 252, 0.55);
        margin-bottom: 0.75rem;
    }
    .section-note {
        font-size: 0.96rem;
        color: #5b6470;
    }
    .pill {
        display: inline-block;
        border-radius: 999px;
        padding: 0.3rem 0.7rem;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 0.35rem;
        margin-bottom: 0.3rem;
        border: 1px solid rgba(49, 51, 63, 0.12);
        background: #f5f7fb;
    }
    .llm-response {
        border-left: 3px solid #4f8ef7;
        padding: 0.75rem 1rem;
        background: rgba(79, 142, 247, 0.05);
        border-radius: 0 8px 8px 0;
        margin-top: 0.5rem;
        font-size: 0.97rem;
        line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def currency(value: float) -> str:
    return f"${value:,.2f}"


def progress_stage(progress: float) -> str:
    pct = int(progress * 100)
    if pct >= 95:
        return "Very late"
    if pct >= 85:
        return "Late"
    if pct >= 50:
        return "Mid"
    return "Early"


def buyer_recommendation(snapshot: Dict[str, Any], quantiles: Dict[str, float], aggressiveness: str) -> Dict[str, Any]:
    threshold_map = {
        "Conservative": quantiles["q50"],
        "Balanced": quantiles["q75"],
        "Aggressive": quantiles["q90"],
    }
    threshold = threshold_map[aggressiveness]
    current_price = snapshot["current_price"]
    decision = "PASS" if current_price >= threshold else "BID"

    return {
        "decision": decision,
        "threshold": round(threshold, 2),
        "current_price": round(current_price, 2),
        "headroom": round(threshold - current_price, 2),
        "aggressiveness": aggressiveness,
    }


def get_snapshot_input(prefix: str) -> Dict[str, Any]:
    col1, col2, col3 = st.columns(3)
    with col1:
        item_name = st.selectbox("Item name", ITEM_OPTIONS, index=1, key=f"{prefix}_item_name")
        auction_type = st.selectbox("Auction type", AUCTION_TYPES, index=2, key=f"{prefix}_auction_type")
        opening_bid = st.number_input(
            "Opening bid",
            min_value=0.0,
            value=25.0,
            step=1.0,
            key=f"{prefix}_opening_bid",
        )
    with col2:
        current_price = st.number_input(
            "Current price",
            min_value=0.0,
            value=60.0,
            step=1.0,
            key=f"{prefix}_current_price",
        )
        auction_progress_pct = st.select_slider(
            "Auction progress (%)",
            options=CHECKPOINTS,
            value=75,
            key=f"{prefix}_progress",
        )
        num_bids_so_far = st.number_input(
            "Number of bids so far",
            min_value=0,
            value=10,
            step=1,
            key=f"{prefix}_num_bids",
        )
    with col3:
        num_unique_bidders_so_far = st.number_input(
            "Unique bidders so far",
            min_value=0,
            value=4,
            step=1,
            key=f"{prefix}_num_bidders",
        )
        highest_observed_bid = st.number_input(
            "Highest observed bid so far",
            min_value=0.0,
            value=max(current_price, opening_bid),
            step=1.0,
            key=f"{prefix}_highest_bid",
        )

    return {
        "label": "Manual input",
        "item_name": item_name,
        "auction_type": auction_type,
        "auction_progress": auction_progress_pct / 100.0,
        "opening_bid": float(opening_bid),
        "current_price": float(current_price),
        "num_bids_so_far": int(num_bids_so_far),
        "num_unique_bidders_so_far": int(num_unique_bidders_so_far),
        "highest_observed_bid": float(highest_observed_bid),
    }


def render_snapshot_summary(snapshot: Dict[str, Any]) -> None:
    stage = progress_stage(snapshot["auction_progress"])
    st.markdown("<div class='auction-card'>", unsafe_allow_html=True)
    st.markdown(f"### {snapshot['item_name']}")
    st.markdown(
        f"<span class='pill'>{snapshot['auction_type']}</span>"
        f"<span class='pill'>{int(snapshot['auction_progress'] * 100)}% progress</span>"
        f"<span class='pill'>{stage} stage</span>"
        f"<span class='pill'>{snapshot['num_bids_so_far']} bids</span>"
        f"<span class='pill'>{snapshot['num_unique_bidders_so_far']} bidders</span>",
        unsafe_allow_html=True,
    )
    col1, col2, col3 = st.columns(3)
    col1.metric("Opening bid", currency(snapshot["opening_bid"]))
    col2.metric("Current price", currency(snapshot["current_price"]))
    col3.metric("Highest observed bid", currency(snapshot["highest_observed_bid"]))
    st.markdown("</div>", unsafe_allow_html=True)


def render_prediction_cards(point_pred: float, quantiles: Dict[str, float]) -> None:
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Point estimate", currency(point_pred))
    col2.metric("q10", currency(quantiles["q10"]))
    col3.metric("q50", currency(quantiles["q50"]))
    col4.metric("q75", currency(quantiles["q75"]))
    col5.metric("q90", currency(quantiles["q90"]))


def render_quantile_range(quantiles: Dict[str, float], current_price: float) -> None:
    st.subheader("Quantile range")
    st.markdown(
        f"Likely outcome band: **{currency(quantiles['q10'])} -> {currency(quantiles['q90'])}**  \n"
        f"Typical / median outcome: **{currency(quantiles['q50'])}**"
    )
    frame = pd.DataFrame(
        {
            "Quantile": ["Current", "q10", "q50", "q75", "q90"],
            "Price": [
                current_price,
                quantiles["q10"],
                quantiles["q50"],
                quantiles["q75"],
                quantiles["q90"],
            ],
        }
    )
    st.dataframe(frame, use_container_width=True, hide_index=True)


def render_buyer_threshold_cards(rec: Dict[str, Any], quantiles: Dict[str, float]) -> None:
    st.subheader("Buyer PASS / threshold cards")
    labels = [
        ("Conservative", quantiles["q50"]),
        ("Balanced", quantiles["q75"]),
        ("Aggressive", quantiles["q90"]),
    ]
    cols = st.columns(3)
    for col, (label, threshold) in zip(cols, labels):
        decision = "PASS" if rec["current_price"] >= threshold else "BID"
        delta = round(threshold - rec["current_price"], 2)
        with col:
            st.markdown(f"### {label}")
            st.metric("Threshold", currency(threshold), delta=currency(delta))
            if decision == "PASS":
                st.error(f"{decision} · current price is above threshold")
            else:
                st.success(f"{decision} · current price is below threshold")

    st.info(
        f"Selected recommendation: **{rec['aggressiveness']} -> {rec['decision']}** "
        f"at threshold **{currency(rec['threshold'])}**."
    )


def render_seller_chart(scenarios_df: pd.DataFrame) -> None:
    st.subheader("Seller scenario chart")
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(
        scenarios_df["Opening bid"],
        scenarios_df["Predicted final price"],
        marker="o",
        linewidth=2,
    )
    ax.set_xlabel("Opening bid")
    ax.set_ylabel("Predicted final price")
    ax.set_title("Opening-bid scenario sweep")
    ax.grid(True, alpha=0.3)

    for _, row in scenarios_df.iterrows():
        ax.annotate(
            row["Scenario"],
            (row["Opening bid"], row["Predicted final price"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
        )

    st.pyplot(fig, clear_figure=True)
    st.dataframe(scenarios_df, use_container_width=True, hide_index=True)


def render_driver_summary(prediction: Dict[str, Any]) -> None:
    st.subheader("Key drivers from this input")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Upside signals**")
        for item in prediction["top_positive_factors"]:
            st.markdown(f"- {item}")
    with col2:
        st.markdown("**Caution signals**")
        for item in prediction["top_negative_factors"]:
            st.markdown(f"- {item}")


def render_explanation(explanation: Dict[str, str]) -> None:
    st.subheader("Explanation + limitations")
    for title, body in explanation.items():
        with st.expander(title, expanded=title == "What the model expects"):
            st.write(body)


def render_llm_panel(
    mode: str,
    snapshot: Dict[str, Any],
    prediction: Dict[str, Any],
    rec: Optional[Dict[str, Any]] = None,
) -> None:
    st.subheader("AI Analysis")
    api_key = st.session_state.get("anthropic_api_key", "")

    cache_key = f"{mode}_llm_cache"
    active_key = f"{mode}_llm_active"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = {}
    if active_key not in st.session_state:
        st.session_state[active_key] = None

    cols = st.columns(4)
    for col, option in zip(cols, LLM_OPTIONS):
        with col:
            if st.button(option, key=f"{mode}_llm_{option}", use_container_width=True):
                st.session_state[active_key] = option
                if option not in st.session_state[cache_key]:
                    if not api_key:
                        st.session_state[cache_key][option] = (
                            "No API key found. Set ANTHROPIC_API_KEY in your terminal before running the app."
                        )
                    else:
                        with st.spinner("Thinking..."):
                            st.session_state[cache_key][option] = get_llm_response(
                                option=option,
                                mode=mode,
                                snapshot=snapshot,
                                prediction=prediction,
                                api_key=api_key,
                                rec=rec,
                            )

    active_option = st.session_state.get(active_key)
    if active_option and active_option in st.session_state[cache_key]:
        raw = st.session_state[cache_key][active_option]
        clean = (
            raw.replace("`", "")
               .replace("**", "")
               .replace("##", "")
               .replace("###", "")
               .replace("*", "")
        )
        with st.container(border=True):
            for para in clean.split("\n\n"):
                para = para.strip()
                if para:
                    st.write(para)


missing_paths = missing_artifact_paths()

st.title("Auction IQ")
st.caption("Snapshot-based buyer and seller decision support powered by Week 9 and Week 11 models.")

with st.sidebar:
    st.header("About this app")
    st.write(
        "This interface uses the trained Week 9 point model and Week 11 quantile models "
        "from the local `models/` directory."
    )
    if missing_paths:
        st.error(
            "Missing model artifacts:\n- " + "\n- ".join(path.name for path in missing_paths)
        )
    else:
        week9_rmse = load_week9_metrics()["overall"]["random_forest_point_model"]["rmse"]
        week11_coverage = load_week11_metrics()["intervals"]["q10_q90"]["coverage"]
        st.success("Loaded point and quantile artifacts from `./models`.")
        st.caption(
            f"Week 9 point model RMSE: {week9_rmse:.1f} | "
            f"Week 11 q10-q90 coverage: {week11_coverage:.1%}"
        )
    st.markdown("**Supported checkpoints:** 25%, 50%, 75%, 85%, 90%, 95%, 100%")
    st.caption("`leading_bidder_rate_so_far` defaults to the training-set median (6.0).")

    st.divider()
    _env_key = os.environ.get("ANTHROPIC_API_KEY", "")
    st.session_state["anthropic_api_key"] = _env_key

if missing_paths:
    st.stop()


buyer_tab, seller_tab = st.tabs(["Buyer", "Seller"])

with buyer_tab:
    st.header("Buyer")
    st.markdown(
        "<p class='section-note'>Use this tab to evaluate whether the current auction price still fits your bidding style.</p>",
        unsafe_allow_html=True,
    )
    buyer_snapshot = get_snapshot_input("buyer")
    render_snapshot_summary(buyer_snapshot)

    aggressiveness = st.selectbox(
        "Bid aggressiveness",
        ["Conservative", "Balanced", "Aggressive"],
        index=1,
    )

    buyer_issues = validate_snapshot(buyer_snapshot)
    if buyer_issues:
        for issue in buyer_issues:
            st.warning(issue)

    if st.button("Run buyer prediction", type="primary", use_container_width=True):
        prediction = predict_snapshot(buyer_snapshot)
        rec = buyer_recommendation(buyer_snapshot, prediction["quantiles"], aggressiveness)
        explanation = build_explanation(buyer_snapshot, prediction, mode="buyer")

        st.session_state["buyer_prediction"] = prediction
        st.session_state["buyer_snapshot"] = buyer_snapshot
        st.session_state["buyer_rec"] = rec
        st.session_state["buyer_explanation"] = explanation
        st.session_state["buyer_llm_cache"] = {}
        st.session_state["buyer_llm_active"] = None

    if st.session_state.get("buyer_prediction"):
        _pred = st.session_state["buyer_prediction"]
        _rec = st.session_state["buyer_rec"]
        _snap = st.session_state["buyer_snapshot"]
        _expl = st.session_state["buyer_explanation"]

        render_prediction_cards(_pred["point_estimate"], _pred["quantiles"])
        render_quantile_range(_pred["quantiles"], _snap["current_price"])
        render_buyer_threshold_cards(_rec, _pred["quantiles"])
        render_driver_summary(_pred)
        render_explanation(_expl)

        st.divider()
        render_llm_panel(
            mode="buyer",
            snapshot=_snap,
            prediction=_pred,
            rec=_rec,
        )

with seller_tab:
    st.header("Seller")
    st.markdown(
        "<p class='section-note'>Use this tab to compare how small changes in opening bid may shift the predicted final price.</p>",
        unsafe_allow_html=True,
    )
    seller_snapshot = get_snapshot_input("seller")
    render_snapshot_summary(seller_snapshot)

    seller_issues = validate_snapshot(seller_snapshot)
    if seller_issues:
        for issue in seller_issues:
            st.warning(issue)

    if st.button("Run seller prediction", type="primary", use_container_width=True):
        prediction = predict_snapshot(seller_snapshot)
        scenarios_df = seller_scenarios(seller_snapshot)
        explanation = build_explanation(seller_snapshot, prediction, mode="seller")

        st.session_state["seller_prediction"] = prediction
        st.session_state["seller_snapshot"] = seller_snapshot
        st.session_state["seller_scenarios_df"] = scenarios_df
        st.session_state["seller_explanation"] = explanation
        st.session_state["seller_llm_cache"] = {}
        st.session_state["seller_llm_active"] = None

    if st.session_state.get("seller_prediction"):
        _pred = st.session_state["seller_prediction"]
        _snap = st.session_state["seller_snapshot"]
        _expl = st.session_state["seller_explanation"]
        _scenarios = st.session_state["seller_scenarios_df"]

        render_prediction_cards(_pred["point_estimate"], _pred["quantiles"])
        render_quantile_range(_pred["quantiles"], _snap["current_price"])
        render_seller_chart(_scenarios)
        render_driver_summary(_pred)
        render_explanation(_expl)

        st.divider()
        render_llm_panel(
            mode="seller",
            snapshot=_snap,
            prediction=_pred,
        )
