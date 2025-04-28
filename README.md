# ⚡ Alberta Electricity Load Forecast & Anomaly Detection

End-to-end pipeline that cleans AESO hourly-load workbooks, trains probabilistic 
**LightGBM** models (P10 / P50 / P90), flags anomalies with **Isolation Forest**, and serves everything in a multi-tab **Streamlit** dashboard.

---

## 🚀 Highlights
- **ETL** – messy Excel → tidy Parquet (200 k+ rows)
- **Forecasting** – quantile LightGBM for every region
- **Anomaly radar** – residual-based Isolation Forest
- **Dashboard** – Plotly charts, KPI strip, dark/light mode, CSV export

---

## 📂 Project layout
src/ ├─ etl_build_dataset.py # data prep ├─ train_forecast.py # model + anomalies └─ streamlit_app.py # Streamlit UI 
data/ └─ clean/ ├─ load_raw.parquet ├─ load_long.parquet └─ preds.parquet

## 🛠 Stack
| Layer | Tools |
|-------|-------|
| Data | **pandas · Parquet** |
| ML   | **LightGBM quantile · scikit-learn** |
| App  | **Streamlit · Plotly** |
| Ops  | **Git / GitHub** |

---

## ⚙️ Quick start
```bash
git clone https://github.com/samira670/load_forecast_project.git
cd load_forecast_project
pip install -r requirements.txt          # pandas, lightgbm, streamlit, …
python src/train_forecast.py             # build preds.parquet (~20 s)
streamlit run src/streamlit_app.py       # open dashboard at localhost:8501
