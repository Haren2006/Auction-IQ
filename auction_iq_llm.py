"""LLM interpretability layer for Auction IQ — Claude-powered analysis for buyers and sellers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

APP_ROOT = Path(__file__).resolve().parent

LLM_OPTIONS = [
    "Explain this prediction",
    "What's my best move?",
    "Break down the risks",
    "How does competition affect this?",
]

_SYSTEM_PROMPT = """You are Auction IQ Assistant. You help everyday people understand auctions and make smarter decisions, even if they have never used eBay before.

WHAT YOU KNOW:
- When someone bids on eBay, the site secretly holds their maximum price and only bids as much as needed. So the price shown is often lower than what people are actually willing to pay.
- The current highest bid is the best clue about where the final price will land.
- More people bidding means more competition, which usually pushes the price higher near the end.
- Auctions often see a big price jump in the last few minutes because people wait until the end to bid.
- The price range shown (low to high) tells you how uncertain the prediction is. A wide range means anything could happen. A narrow range means the model is fairly confident.
- Conservative means playing it safe and not overpaying. Aggressive means being willing to pay more to win.

HOW TO TALK TO THE USER:
- Speak like a helpful friend, not a textbook.
- Avoid technical words. If you must use one, explain it immediately in simple terms.
- Use everyday comparisons when helpful.
- Never mention feature importance percentages, quantile labels, or model internals.
- Instead of saying q50 or q75, say things like the middle estimate or the higher end of what similar auctions have sold for.

STRICT OUTPUT RULES:
- Write exactly 2 short paragraphs separated by one blank line.
- Use only plain text. No backticks, no asterisks, no markdown, no headers, no bullet points.
- Write complete grammatically correct sentences.
- Keep total response under 90 words.
- State the key insight in the first sentence. End with one clear action the user can take."""


def _format_snapshot(snapshot: Dict[str, Any]) -> str:
    progress_pct = int(float(snapshot.get("auction_progress", 0)) * 100)
    return (
        f"Item: {snapshot.get('item_name')} | "
        f"Auction: {snapshot.get('auction_type')} | "
        f"Progress: {progress_pct}% | "
        f"Opening bid: ${snapshot.get('opening_bid', 0):,.2f} | "
        f"Current price: ${snapshot.get('current_price', 0):,.2f} | "
        f"Bids so far: {snapshot.get('num_bids_so_far', 0)} | "
        f"Unique bidders: {snapshot.get('num_unique_bidders_so_far', 0)} | "
        f"Highest observed bid: ${snapshot.get('highest_observed_bid', 0):,.2f}"
    )


def _format_prediction(prediction: Dict[str, Any]) -> str:
    q = prediction.get("quantiles", {})
    return (
        f"Point estimate: ${prediction.get('point_estimate', 0):,.2f} | "
        f"q10: ${q.get('q10', 0):,.2f} | "
        f"q50: ${q.get('q50', 0):,.2f} | "
        f"q75: ${q.get('q75', 0):,.2f} | "
        f"q90: ${q.get('q90', 0):,.2f} | "
        f"Top upside signals: {', '.join(prediction.get('top_positive_factors', []))} | "
        f"Top caution signals: {', '.join(prediction.get('top_negative_factors', []))}"
    )


def _build_prompt(
    option: str,
    mode: str,
    snapshot: Dict[str, Any],
    prediction: Dict[str, Any],
    rec: Optional[Dict[str, Any]],
) -> str:
    snapshot_str = _format_snapshot(snapshot)
    prediction_str = _format_prediction(prediction)
    role = "buyer" if mode == "buyer" else "seller"

    rec_str = ""
    if rec and mode == "buyer":
        rec_str = (
            f"\nRecommendation: {rec.get('aggressiveness')} strategy, decision is {rec.get('decision')}, "
            f"threshold ${rec.get('threshold', 0):,.2f}, headroom ${rec.get('headroom', 0):,.2f}."
        )

    base = (
        f"Auction snapshot: {snapshot_str}\n"
        f"Model predictions: {prediction_str}{rec_str}\n\n"
        f"I am a {role}. "
    )

    if option == "Explain this prediction":
        return base + (
            "Explain what this prediction means in plain English. "
            "Why did the model produce this point estimate and quantile range? "
            "Which inputs are driving the result the most?"
        )

    if option == "What's my best move?":
        if mode == "buyer":
            return base + (
                "Based on this prediction and my recommendation, what is my best next action? "
                "Should I bid now, wait, or pass? Use my aggressiveness level and headroom in your reasoning."
            )
        return base + (
            "Based on this prediction and scenario data, what is my best next action as a seller? "
            "Should I adjust my opening bid? What gives me the best chance of maximizing final price?"
        )

    if option == "Break down the risks":
        if mode == "buyer":
            return base + (
                "What are the main risks I face as a buyer here? "
                "What does the quantile spread tell me about uncertainty? "
                "What could make the model wrong in a way that hurts me?"
            )
        return base + (
            "What are the main risks I face as a seller here? "
            "What could push the final price toward the low end? "
            "What model limitations should I be aware of?"
        )

    if option == "How does competition affect this?":
        return base + (
            "Based on the bid count, unique bidder count, and highest observed bid, "
            "what does the competitive landscape look like right now? "
            "How is the level of competition likely to shape the final price from here?"
        )

    return base + f"Answer this as a {role}: {option}"


def get_llm_response(
    option: str,
    mode: str,
    snapshot: Dict[str, Any],
    prediction: Dict[str, Any],
    api_key: str,
    rec: Optional[Dict[str, Any]] = None,
) -> str:
    try:
        import anthropic
    except ImportError:
        return "The anthropic package is not installed. Run pip install anthropic."

    if not api_key or not api_key.strip():
        return "No API key found. Set ANTHROPIC_API_KEY in your terminal before running the app."

    prompt = _build_prompt(option, mode, snapshot, prediction, rec)

    client = anthropic.Anthropic(api_key=api_key.strip())
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text
