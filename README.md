# Auction IQ  
### Predictive Decision Support for Online Auction Buyers and Sellers

## Overview
Auction IQ is an academic machine learning and decision-support project designed to help users interpret live online auction conditions through a snapshot-based prediction interface. The system estimates likely final auction prices using intermediate auction information and presents the results through an interactive Streamlit dashboard.

The project is structured around two user perspectives:

- **Buyer view**: helps a bidder decide whether the current auction price is still worth pursuing
- **Seller view**: helps a seller evaluate how small changes in the opening bid may influence the predicted final sale price

Rather than focusing only on a single final estimate, Auction IQ emphasizes interpretable decision support through prediction intervals, threshold logic, scenario analysis, and clear explanations of uncertainty.

---

## Project Objective
The objective of this project is to build an end-to-end interface that translates auction snapshot data into practical recommendations for buyers and sellers. The application supports manual and example-based inputs and allows users to explore model outputs in a way that is visually clear, academically grounded, and easy to interpret.

---

## Key Features

### Buyer Tab
The Buyer tab is designed to support bidding decisions by showing:

- a **point prediction** for the expected final auction price
- a **quantile range** to reflect uncertainty in the predicted outcome
- **threshold-based recommendation cards**
  - Conservative → based on q50
  - Balanced → based on q75
  - Aggressive → based on q90
- a **PASS / BID recommendation** depending on whether the current price exceeds the selected threshold

### Seller Tab
The Seller tab is designed to support listing strategy by showing:

- the same core prediction outputs
- a **seller scenario chart**
- a comparison of opening-bid adjustments at:
  - -20%
  - -10%
  - current
  - +10%
  - +20%

This helps illustrate how changes in the opening bid may affect the predicted final sale price.

### Input Modes
The interface supports two modes of use:

- **Example mode**: loads predefined sample auction snapshots
- **Manual mode**: allows a user to enter auction values directly

### Explanation Layer
To improve interpretability, the interface includes:

- what the model expects
- why the prediction was generated
- suggested action
- model limitations

---

## Methodology
Auction IQ uses a snapshot-based prediction framework. Instead of waiting for the auction to end, the system uses intermediate auction information to estimate the likely final price. The current implementation is structured to support integration with saved machine learning models and quantile estimators.

The Streamlit interface is designed as a front-end decision-support layer that can connect to:

- a point prediction model
- quantile prediction outputs
- buyer threshold logic
- seller scenario analysis
- explanation and limitation summaries

---

## Input Variables
The application currently uses the following auction snapshot inputs:

- item name
- auction progress
- opening bid
- current price
- number of bids so far
- number of unique bidders so far
- highest observed bid so far

These variables are used to simulate live auction conditions and produce predictive outputs for both the buyer and seller workflows.

---

## User Interface Design
The Streamlit application is organized into two main tabs for clarity and usability. The design focuses on readability, interpretability, and practical decision support rather than only raw prediction output.

The interface includes:

- structured snapshot summary cards
- prediction metric cards
- quantile tables
- buyer threshold cards
- seller scenario visualization
- explanation and limitations sections

This design allows the system to function not only as a predictive tool, but also as an academic demonstration of human-centered ML interface design.

---

## How to Run the App

### 1. Install dependencies
```bash
pip install -r requirements.txt
