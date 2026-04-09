# Auction IQ

Auction IQ is a Streamlit decision-support app for live auction snapshots. This repo is set up so teammates can clone it, install dependencies, and run the app without retraining the Week 9 point model or the Week 11 quantile models.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Note: the committed model artifacts were exported with `scikit-learn 1.6.1`, so the runtime requirements pin that version for compatibility.

The app expects these committed artifacts in `models/`:

- `point_model.pkl`
- `metadata.json`
- `week9_metrics.json`
- `quantile_models.pkl`
- `week11_quantile_metadata.json`
- `week11_quantile_metrics.json`

## Project Layout

- `streamlit_app.py`: canonical Streamlit entrypoint
- `auction_iq_backend.py`: model-loading and UI-to-model mapping adapter
- `models/`: committed Week 9 point-model and Week 11 quantile-model artifacts
- `src/week9/`: training and point-prediction helpers
- `notebooks/`: Week 9 and Week 11 training notebooks plus EDA notebooks

## Runtime Notes

- Teammates do not need `auction.csv` or `auction_snapshots.csv` to run the app. Those files are only useful for retraining or analysis.
- The Streamlit UI maps user inputs into the model schema expected by the trained artifacts.
- `auction_type` is limited to `3 day auction`, `5 day auction`, and `7 day auction`.
- `leading_bidder_rate_so_far` defaults to the Week 9 training-set median of `6.0`.
- Quantiles are post-processed so the app always returns `q10 <= q50 <= q75 <= q90`.

## Training Provenance

The notebooks in `notebooks/` are kept for provenance:

- Week 9 snapshot point-model training
- Week 11 quantile-model training
- supporting exploratory analysis

If you retrain the models, overwrite the files in `models/` with the new exported artifacts before sharing the branch.
