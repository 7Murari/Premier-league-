# ⚽ Premier League Player Valuation Engine

An end-to-end Machine Learning web application and interactive Streamlit dashboard that predicts Premier League player transfer valuations. The engine merges multi-year Opta performance data from **FBref** with real-world ground-truth market values from **Transfermarkt**.

---

## 📌 Project Overview

Traditional football valuation models often over-index on goals and assists, resulting in inflated prices for offensive players and inaccurate estimates for defenders, deep-lying playmakers, and aging veterans.

This engine addresses those limitations by implementing:
* **Position-Aware Metrics:** Tailored feature evaluation across Forwards, Midfielders, and Defenders (incorporating tackles, interceptions, clearances, and playing durability).
* **Non-Linear Age Decay:** Exponential resale multipliers and age caps to prevent runaway valuations for older veterans while preserving premium pricing for young prospects and prime stars.
* **Annualized Performance Pace:** Normalizes player statistics across actual active seasons to compare multi-year Premier League veterans fairly against recent arrivals.
* **Transfermarkt Calibration:** Trains on real-world consensus market values mapped directly to Opta performance metrics via Ridge Regression.

---

## 🛠️ Tech Stack

* **Language:** Python
* **Web Framework:** Streamlit
* **Machine Learning:** Scikit-Learn (`Ridge` Regression)
* **Data Scraping & ETL:** `soccerdata` (FBref), `pandas`, `numpy`
* **Entity Matching:** `rapidfuzz` (Fuzzy string matching for player alignment)
* **Visualization:** Matplotlib

---

## 📁 Project Structure

```text
.
├── app.py                      # Interactive Streamlit dashboard
├── build_dataset.py            # Automated multi-table FBref & Transfermarkt ETL pipeline
├── pl_players_3yr_complete.csv # Local cached multi-year player dataset
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation


