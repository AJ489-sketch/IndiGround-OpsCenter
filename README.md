# ✈️ IndiGround OpsCenter — Setup & Deployment Guide

## Your Project File Structure

```
indiground/
├── app.py                    ← Main Streamlit app
├── requirements.txt          ← Python dependencies
├── tat_model.pkl             ← Your trained XGBoost TAT model  ← ADD THIS
├── delay_model.pkl           ← Your trained delay classifier   ← ADD THIS
├── baggage_flow.csv          ← Historical data                 ← ADD THIS
├── catering_logs.csv         ← Historical data                 ← ADD THIS
├── fuel_operations.csv       ← Historical data                 ← ADD THIS
└── .streamlit/
    └── config.toml           ← Dark theme config
```

---

## STEP-BY-STEP INSTRUCTIONS

### STEP 1 — Install Python dependencies locally

Open your terminal and run:

```bash
pip install streamlit pandas numpy plotly xgboost scikit-learn joblib statsmodels
```

---

### STEP 2 — Organize your files

Create a folder called `indiground` on your computer.
Copy into it:
- `app.py` (this file)
- `requirements.txt`
- `.streamlit/config.toml` (create this subfolder)
- `tat_model.pkl` (your trained model from Colab)
- `delay_model.pkl` (your trained model from Colab)
- `baggage_flow.csv`
- `catering_logs.csv`
- `fuel_operations.csv`

---

### STEP 3 — Test locally

In your terminal, navigate into the folder and run:

```bash
cd indiground
streamlit run app.py
```

It will open at http://localhost:8501
Check all 3 pages work correctly.

---

### STEP 4 — Push to GitHub

If you don't have git installed: https://git-scm.com/downloads

```bash
git init
git add .
git commit -m "IndiGround OpsCenter - Hackathon build"
```

Create a NEW repository on github.com (call it `indiground-ops`)
Then push:

```bash
git remote add origin https://github.com/YOUR_USERNAME/indiground-ops.git
git branch -M main
git push -u origin main
```

---

### STEP 5 — Deploy on Streamlit Community Cloud (FREE)

1. Go to https://share.streamlit.io
2. Sign in with your GitHub account
3. Click **"New app"**
4. Select:
   - Repository: `indiground-ops`
   - Branch: `main`
   - Main file path: `app.py`
5. Click **"Deploy!"**

Your live URL will be:
`https://YOUR_USERNAME-indiground-ops-app-XXXX.streamlit.app`

Deployment takes ~2 minutes. Share this URL with judges!

---

## DEMO SCRIPT (for the presentation)

1. Open **Live Gate Dashboard** → Show 6 active gates with color-coded risk
2. Point to a RED gate → Show the ₹ penalty and bottleneck recommendation
3. Click **Refresh Gates** → Demonstrate real-time re-prediction
4. Switch to **Flight Predictor** → Enter a high-risk flight (lots of special meals, FAIL safety)
5. Show the radar chart + resource recommendation
6. Switch to **Historical Insights** → Show the model metrics (R²=0.9638) and key findings
7. Closing line: *"Our model predicted TAT within 7.8 minutes on average — deployed across 500 flights, this system would have saved ₹X in delay penalties."*

---

## IMPORTANT NOTES

- The app works even WITHOUT the .pkl files (it runs in demo/simulation mode)
- If models ARE present, it uses real XGBoost predictions
- The CSV files are needed for the Historical Insights page real charts
- Without CSVs, the page shows simulated data that still looks correct
