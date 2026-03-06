# Logistics Control Tower & Routing Optimizer

https://logistics-control-tower-5yclyyiwvd5an78wbnq3rt.streamlit.app/

## Overview
This application is a Python-based Logistics Control Tower designed to optimize LTL (Less-Than-Truckload) to FTL (Full-Truckload) freight consolidation. It evaluates multi-node retail demand and dynamically pools inventory into FTL shipments based on physical, financial, and regulatory constraints. 

The engine incorporates financial arbitrage calculations, Monte Carlo risk simulations, and automated compliance checks to ensure all generated routes are legally and physically viable before generating a manifest.

## Core Features
* **Capacity Engine:** Evaluates payloads against trailer cube (cu ft), DOT weight limits (lbs), and floor pallet positions, including double-stacking constraints.
* **Stochastic Risk Modeling:** Runs 1,000-iteration Monte Carlo simulations against dynamic transit times to calculate network survival probability and Expected Shortage Cost (ESC).
* **Demand Forecasting:** Uses ARIMA time-series modeling to project 7-day inventory depletion across the retail network.
* **Regulatory Clearance:** Algorithmically enforces FMCSA 14-Hour Driver HOS rules, DOT weigh station limits, and retail backroom footprint constraints.
* **Financial Arbitrage:** Calculates total landed cost by comparing LTL base rates against FTL linehauls, dynamically applying accessorials (detention, liftgate, lumper fees) and holding cost penalties (WACC, inventory spoilage).
* **TMS Integration:** Exports machine-readable JSON payload manifests for direct tender into external Transportation Management Systems.

## Tech Stack
* **Frontend:** Streamlit
* **Data & Math:** Pandas, NumPy, Statsmodels (ARIMA)
* **Routing API:** OSRM (Open Source Routing Machine)

## Local Setup
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `streamlit run dashboard.py`
