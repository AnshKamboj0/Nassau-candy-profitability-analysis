# Nassau Candy – Product Profitability & Margin Intelligence

An interactive Streamlit analytics and machine-learning dashboard for the Nassau Candy Distributor dataset.

## Features
- Executive KPIs: revenue, cost, gross profit, gross margin and units
- Monthly revenue and profit trends
- Product profitability leaderboard
- Division-level revenue and margin comparison
- K-Means product segmentation
- Isolation Forest anomaly screening
- Revenue and profit Pareto analysis
- State-level performance and factory coordinate map
- CSV downloads for product and ML outputs

## Dataset
Place `Nassau Candy Distributor.csv` in this folder, or upload it through the dashboard sidebar.

Expected columns include:
`Sales`, `Units`, `Cost`, `Product Name`, `Division`, and optionally `Gross Profit`, `Order Date`, `Ship Date`, and `State/Province`.

## Setup
```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Run:
```bash
streamlit run app.py
```

## Important data-quality observation
The supplied file has 10,194 rows and 18 columns. Its order dates range from 2024-01-02 to 2025-12-31, while ship dates range from 2026-06-30 to 2030-06-28. This creates unusually long shipping lags (about 1,321 days on average). Confirm the date fields with the dataset provider before drawing conclusions about shipping performance or delivery efficiency.

## Machine-learning interpretation
K-Means is used for exploratory product segmentation, not supervised prediction. Isolation Forest flags statistically unusual product profiles for investigation; a flag does not prove an error or misconduct. With only 15 distinct products in this dataset, treat model findings as exploratory and interpret them alongside business metrics.

## Business metric
Gross Margin (%) = Total Gross Profit / Total Sales × 100.

Product-level metrics are aggregated from transaction rows before ranking.
