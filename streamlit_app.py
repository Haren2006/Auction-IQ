import math
from typing import Dict, Any, List

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Auction IQ",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


EXAMPLE_ROWS = [
    {
        "label": "Palm Pilot · 50% progress",
        "item_name": "Palm Pilot M515 PDA",
        "auction_progress": 0.50,
        "opening_bid": 25.0,
        "current_price": 72.0,
        "num_bids_so_far": 9,
        "num_unique_bidders_so_far": 4,
        "highest_observed_bid": 72.0,
    },
    {
        "label": "Xbox · 85% progress",
        "item_name": "Xbox game console",
        "auction_progress": 0.85,
        "opening_bid": 40.0,
        "current_price": 118.0,
        "num_bids_so_far": 14,
        "num_unique_bidders_so_far": 6,
        "highest_observed_bid": 118.0,
    },
    {
        "label": "Cartier · 90% progress",
        "item_name": "Cartier wristwatch",
        "auction_progress": 0.90,
        "opening_bid": 500.0,
        "current_price": 1850.0,
        "num_bids_so_far": 18,
        "num_unique_bidders_so_far": 8,
        "highest_observed_bid": 1850.0,
    },
]

CHECKPOINTS = [25, 50, 75, 85, 90, 95]


# =========================
# Styling
# =========================
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
    </style>
    """,
    unsafe_allow_html=True,
)


def predict_point(snapshot: Dict[str, Any]) -> float:
    progress = snapshot["auction_progress"]
    current_price = snapshot["current_price"]
    n_bids = snapshot["num_bids_so_far"]
    n_bidders = snapshot["num_unique_bidders_so_far"]
    opening_bid = snapshot["opening_bid"]
    highest_observed_bid = snapshot["highest_observed_bid"]

    growth_factor = 1.0 + (1.15 * (1.0 - progress))
    bid_signal = 1.0 + min(n_bids, 25) * 0.012 + min(n_bidders, 12) * 0.018
    competition_signal = 1.0 + min(max(highest_observed_bid - current_price, 0), 500) / 10000
    opening_signal = 0.08 * opening_bid

    pred = current_price * growth_factor * bid_signal * competition_signal + opening_signal
    return round(max(pred, current_price), 2)


def predict_quantiles(snapshot: Dict[str, Any], point_pred: float | None = None) -> Dict[str, float]:
    if point_pred is None:
        point_pred = predict_point(snapshot)

    progress = snapshot["auction_progress"]
    spread = max(0.08, 0.28 - 0.18 * progress)

    q10 = max(snapshot["current_price"], point_pred * (1 - spread))
    q50 = max(snapshot["current_price"], point_pred * (1 - spread * 0.30))
    q75 = max(snapshot["current_price"], point_pred * (1 + spread * 0.20))
    q90 = max(snapshot["current_price"], point_pred * (1 + spread * 0.45))

    ordered = sorted([q10, q50, q75, q90])
    return {
        "q10": round(ordered[0], 2),
        "q50": round(ordered[1], 2),
        "q75": round(ordered[2], 2),
        "q90": round(ordered[3], 2),
    }


def buyer_recommendation(snapshot: Dict[str, Any], quantiles: Dict[str, float], aggressiveness: str) -> Dict[str, Any]:
    threshold_map = {
        "Conservative": quantiles["q50"],
        "Balanced": quantiles["q75"],
        "Aggressive": quantiles["q90"],
    }
    threshold = threshold_map[aggressiveness]
    current_price = snapshot["current_price"]
    decision = "PASS" if current_price >= threshold else "BID"
    headroom = round(threshold - current_price, 2)

    return {
        "decision": decision,
        "threshold": round(threshold, 2),
        "current_price": round(current_price, 2),
        "headroom": headroom,
        "aggressiveness": aggressiveness,
    }


def seller_scenarios(snapshot: Dict[str, Any]) -> pd.DataFrame:
    base = snapshot.copy()
    scenarios = []
    for pct in [-0.20, -0.10, 0.0, 0.10, 0.20]:
        modified = base.copy()
        modified["opening_bid"] = max(0.01, base["opening_bid"] * (1 + pct))
        pred = predict_point(modified)
        scenarios.append(
            {
                "Scenario": f"{pct:+.0%}",
                "Opening bid": round(modified["opening_bid"], 2),
                "Predicted final price": round(pred, 2),
                "Change vs current setup": round(pred - predict_point(base), 2),
            }
        )
    return pd.DataFrame(scenarios)


def driver_summary(snapshot: Dict[str, Any]) -> List[str]:
    signals = []
    progress_pct = int(snapshot["auction_progress"] * 100)
    if progress_pct >= 85:
        signals.append("Late-stage snapshot gives stronger signal")
    else:
        signals.append("Earlier snapshot means more uncertainty")

    if snapshot["num_bids_so_far"] >= 12:
        signals.append("High bidding activity so far")
    else:
        signals.append("Moderate bidding activity so far")

    if snapshot["num_unique_bidders_so_far"] >= 5:
        signals.append("More bidder competition detected")
    else:
        signals.append("Limited bidder competition so far")

    if snapshot["highest_observed_bid"] > snapshot["current_price"]:
        signals.append("Highest observed bid suggests room above current price")
    else:
        signals.append("Highest observed bid is close to current price")

    return signals


def build_explanation(snapshot: Dict[str, Any], point_pred: float, quantiles: Dict[str, float], mode: str) -> Dict[str, str]:
    progress_pct = int(snapshot["auction_progress"] * 100)
    n_bids = snapshot["num_bids_so_far"]
    n_bidders = snapshot["num_unique_bidders_so_far"]

    what_model_expects = (
        f"The model expects later snapshots to be more informative. This input is at {progress_pct}% progress, "
        f"with {n_bids} bids and {n_bidders} unique bidders so far."
    )

    why = (
        f"Current price, opening bid, bid count, unique bidders, highest observed bid, and auction progress are driving the estimate. "
        f"The point prediction is ${point_pred:,.2f} and the likely range runs from q10 ${quantiles['q10']:,.2f} to q90 ${quantiles['q90']:,.2f}."
    )

    if mode == "buyer":
        suggested_action = (
            f"Use the aggressiveness thresholds as guardrails. Conservative uses q50, balanced uses q75, and aggressive uses q90. "
            f"If the live price is already above your chosen threshold, the recommendation becomes PASS."
        )
    else:
        suggested_action = (
            "Use the seller scenario sweep to compare slightly lower or higher opening bids. Treat the chart as directional strategy guidance, not a guarantee."
        )

    limitations = (
        "This UI cannot see listing quality, seller reputation, photos, description quality, sniping, outside demand shocks, or bidder intent. Early-stage snapshots are less certain than late-stage snapshots."
    )

    return {
        "What the model expects": what_model_expects,
        "Why": why,
        "Suggested action": suggested_action,
        "What this model cannot know": limitations,
    }



def currency(x: float) -> str:
    return f"${x:,.2f}"


def progress_stage(progress: float) -> str:
    pct = int(progress * 100)
    if pct >= 95:
        return "Very late"
    if pct >= 85:
        return "Late"
    if pct >= 50:
        return "Mid"
    return "Early"


def validate_snapshot(snapshot: Dict[str, Any]) -> List[str]:
    issues = []
    if snapshot["current_price"] < snapshot["opening_bid"]:
        issues.append("Current price is below opening bid. Check the input values.")
    if snapshot["highest_observed_bid"] < snapshot["current_price"]:
        issues.append("Highest observed bid should usually be at least the current price.")
    if snapshot["num_unique_bidders_so_far"] > snapshot["num_bids_so_far"]:
        issues.append("Unique bidders cannot exceed total bids.")
    return issues


def get_snapshot_input(prefix: str) -> Dict[str, Any]:
    mode = st.radio(
        "Input mode",
        ["Example mode", "Manual mode"],
        horizontal=True,
        key=f"{prefix}_mode",
    )

    if mode == "Example mode":
        labels = [row["label"] for row in EXAMPLE_ROWS]
        selected_label = st.selectbox("Choose a sample row", labels, key=f"{prefix}_example")
        selected = next(row for row in EXAMPLE_ROWS if row["label"] == selected_label)
        st.caption("Loaded from saved sample rows.")
        return dict(selected)

    col1, col2, col3 = st.columns(3)
    with col1:
        item_name = st.text_input("Item name", value="Auction item", key=f"{prefix}_item_name")
        opening_bid = st.number_input("Opening bid", min_value=0.0, value=25.0, step=1.0, key=f"{prefix}_opening_bid")
        current_price = st.number_input("Current price", min_value=0.0, value=60.0, step=1.0, key=f"{prefix}_current_price")
    with col2:
        auction_progress_pct = st.select_slider(
            "Auction progress (%)",
            options=CHECKPOINTS,
            value=75,
            key=f"{prefix}_progress",
        )
        num_bids_so_far = st.number_input("Number of bids so far", min_value=0, value=10, step=1, key=f"{prefix}_num_bids")
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
        "auction_progress": auction_progress_pct / 100.0,
        "opening_bid": float(opening_bid),
        "current_price": float(current_price),
        "num_bids_so_far": int(num_bids_so_far),
        "num_unique_bidders_so_far": int(num_unique_bidders_so_far),
        "highest_observed_bid": float(highest_observed_bid),
    }


def render_snapshot_summary(snapshot: Dict[str, Any]):
    stage = progress_stage(snapshot["auction_progress"])
    st.markdown("<div class='auction-card'>", unsafe_allow_html=True)
    st.markdown(f"### {snapshot['item_name']}")
    st.markdown(
        f"<span class='pill'>{int(snapshot['auction_progress'] * 100)}% progress</span>"
        f"<span class='pill'>{stage} stage</span>"
        f"<span class='pill'>{snapshot['num_bids_so_far']} bids</span>"
        f"<span class='pill'>{snapshot['num_unique_bidders_so_far']} bidders</span>",
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Opening bid", currency(snapshot["opening_bid"]))
    c2.metric("Current price", currency(snapshot["current_price"]))
    c3.metric("Highest observed bid", currency(snapshot["highest_observed_bid"]))
    st.markdown("</div>", unsafe_allow_html=True)


def render_prediction_cards(point_pred: float, quantiles: Dict[str, float]):
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Point estimate", currency(point_pred))
    c2.metric("q10", currency(quantiles["q10"]))
    c3.metric("q50", currency(quantiles["q50"]))
    c4.metric("q75", currency(quantiles["q75"]))
    c5.metric("q90", currency(quantiles["q90"]))


def render_quantile_range(quantiles: Dict[str, float], current_price: float):
    st.subheader("Quantile range")
    st.markdown(
        f"Likely outcome band: **{currency(quantiles['q10'])} → {currency(quantiles['q90'])}**  \\n"
        f"Typical / median outcome: **{currency(quantiles['q50'])}**"
    )

    q_df = pd.DataFrame(
        {
            "Quantile": ["Current", "q10", "q50", "q75", "q90"],
            "Price": [current_price, quantiles["q10"], quantiles["q50"], quantiles["q75"], quantiles["q90"]],
        }
    )
    st.dataframe(q_df, use_container_width=True, hide_index=True)



def render_buyer_threshold_cards(rec: Dict[str, Any], quantiles: Dict[str, float]):
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
        f"Selected recommendation: **{rec['aggressiveness']} → {rec['decision']}** at threshold **{currency(rec['threshold'])}**."
    )



def render_seller_chart(scenarios_df: pd.DataFrame):
    st.subheader("Seller scenario chart")
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(scenarios_df["Opening bid"], scenarios_df["Predicted final price"], marker="o", linewidth=2)
    ax.set_xlabel("Opening bid")
    ax.set_ylabel("Predicted final price")
    ax.set_title("Opening-bid scenario sweep")
    ax.grid(True, alpha=0.3)

    for _, row in scenarios_df.iterrows():
        ax.annotate(row["Scenario"], (row["Opening bid"], row["Predicted final price"]), textcoords="offset points", xytext=(0, 8), ha="center")

    st.pyplot(fig, clear_figure=True)
    st.dataframe(scenarios_df, use_container_width=True, hide_index=True)



def render_driver_summary(snapshot: Dict[str, Any]):
    st.subheader("Key drivers from this input")
    for item in driver_summary(snapshot):
        st.markdown(f"- {item}")



def render_explanation(expl: Dict[str, str]):
    st.subheader("Explanation + limitations")
    for title, body in expl.items():
        with st.expander(title, expanded=True if title == "What the model expects" else False):
            st.write(body)


st.title("Auction IQ")
st.caption("Week 13 UI integration for buyer and seller decision support.")

with st.sidebar:
    st.header("About this UI")
    st.write(
        "This interface is built around snapshot-based auction inputs, buyer quantile thresholds, and seller opening-bid scenarios."
    )
    st.info(
        "When the real backend is ready, replace the fallback functions at the top with your saved point model, quantile models, and explanation layer."
    )
    st.markdown("**Current checkpoints used in the UI:** 25%, 50%, 75%, 85%, 90%, 95%")

buyer_tab, seller_tab = st.tabs(["Buyer", "Seller"])

with buyer_tab:
    st.header("Buyer")
    st.markdown("<p class='section-note'>Use this tab to evaluate whether the current auction price still fits your bidding style.</p>", unsafe_allow_html=True)
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
        point_pred = predict_point(buyer_snapshot)
        quantiles = predict_quantiles(buyer_snapshot, point_pred)
        rec = buyer_recommendation(buyer_snapshot, quantiles, aggressiveness)
        expl = build_explanation(buyer_snapshot, point_pred, quantiles, mode="buyer")

        render_prediction_cards(point_pred, quantiles)
        render_quantile_range(quantiles, buyer_snapshot["current_price"])
        render_buyer_threshold_cards(rec, quantiles)
        render_driver_summary(buyer_snapshot)
        render_explanation(expl)

with seller_tab:
    st.header("Seller")
    st.markdown("<p class='section-note'>Use this tab to compare how small changes in opening bid may shift the predicted final price.</p>", unsafe_allow_html=True)
    seller_snapshot = get_snapshot_input("seller")
    render_snapshot_summary(seller_snapshot)

    seller_issues = validate_snapshot(seller_snapshot)
    if seller_issues:
        for issue in seller_issues:
            st.warning(issue)

    if st.button("Run seller prediction", type="primary", use_container_width=True):
        point_pred = predict_point(seller_snapshot)
        quantiles = predict_quantiles(seller_snapshot, point_pred)
        scenarios_df = seller_scenarios(seller_snapshot)
        expl = build_explanation(seller_snapshot, point_pred, quantiles, mode="seller")

        render_prediction_cards(point_pred, quantiles)
        render_quantile_range(quantiles, seller_snapshot["current_price"])
        render_seller_chart(scenarios_df)
        render_driver_summary(seller_snapshot)
        render_explanation(expl)
