# 🏏 T20 World Cup 2026 AI Intelligence Dashboard

A high-fidelity cricket tournament simulator and match predictor built with **XGBoost Machine Learning** and **Streamlit**. This app predicts match outcomes for the upcoming 2026 T20 World Cup and simulates the entire tournament from the Group Stage to the Grand Final.

## 🚀 Live Demo
[https://t20-worldcup-predictor.streamlit.app]

## ✨ Key Features
* **Match Predictor:** Real-time win probability analysis for any two teams at specific venues.
* **Full Tournament Simulation:** Simulates 40+ group stage matches, tracks the points table, and identifies the **Top 8**.
* **Dynamic Knockouts:** Automatically generates Super 8 groups and predicts the road to the Final.
* **AI Tier System:** Uses a custom "Giant Killer" override logic to ensure realistic results when top-tier teams (India, Australia) face lower-tier nations.

## 🧠 The "Secret Sauce": Technical Approach
Unlike simple simulators, this project uses a two-layered intelligence approach:

1.  **ML Model:** An **XGBoost Classifier** trained on historical T20I data, considering team strength, ICC ratings, and venue history.
2.  **Tiered Logic:** A custom domain-knowledge layer that categorizes teams into 4 tiers. This prevents "AI hallucinations" and ensures that Tier-1 giants maintain their statistical edge against Tier-4 underdogs.

## 🛠️ Tech Stack
* **Python 3.10+**
* **XGBoost:** Gradient boosting for high-accuracy predictions.
* **Streamlit:** For the interactive dashboard UI.
* **Joblib:** For efficient model serialization.
* **Pandas/NumPy:** Data manipulation and feature engineering.

## 📂 Repository Structure
* `app.py`: The main Streamlit dashboard.
* `t20_model.joblib`: The trained XGBoost brain.
* `wc_2026_schedule.csv`: Official fixture data for the simulation.
* `requirements.txt`: Necessary libraries for deployment.

## ⚙️ Local Setup
1. Clone the repo:
   ```bash
   git clone [https://github.com/YourUsername/your-repo-name.git](https://github.com/YourUsername/your-repo-name.git)
2.Install dependencies:
  pip install -r requirements.txt
3.Run the app:
 streamlit run app.py
