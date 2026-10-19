# 🏦 BankGuard AI: Enterprise Retail Credit Risk & Default Prediction

An end-to-end Machine Learning Underwriting Engine and Model Monitoring System for retail consumer loans. Built with regularized **XGBoost**, domain-driven **104 Feature Engineering**, **Cost-Sensitive Threshold Tuning**, and **Population Stability Index (PSI)** monitoring.

---

## 🚀 Live Demo & Web Portal
Interactive Streamlit application features:
- **Instant Credit Scoring:** Real-time Probability of Default (PD) scoring with auto-approve, manual review, and decline badges.
- **Batch Underwriting:** Bulk CSV scoring with downloadable risk-stratified portfolio.
- **Model Governance:** Decile Lift analysis (2.18x lift in Decile 1) and 10-bin PSI stability tracker.
- **Adverse Action Reason Codes:** Basel / FCRA regulatory compliance explanation for declined loans.

---

## 📊 Model Highlights & Performance Benchmarks
- **Dataset:** 15,223 retail loan applications across 52 raw attributes.
- **Class Imbalance:** 88.1% Non-defaulters vs 11.9% Defaulters (1:7.4 ratio).
- **Holdout Test ROC-AUC:** `0.6791` (67.91%)
- **Gini Coefficient:** `0.3582` (35.82%)
- **Kolmogorov-Smirnov (KS) Statistic:** `28.48%`
- **Cost-Optimized Cutoff:** `0.37` (Maximizes defaulter recall to **82.04%**).
- **Population Stability Index (PSI):** `0.00358` (Status: **GREEN / Highly Stable**).

---

## 🛠️ Project Structure
```
├── app.py                           # Main Streamlit web application
├── banking_credit_risk_pipeline.py  # Full training & evaluation pipeline (Spyder ready)
├── credit_risk_artifacts.pkl        # Serialized model, 104-feature maps & encoders
├── banking_model_evaluation.png     # ROC, KS Separation & Cost Curves
├── all_features_iv_vif_report.csv   # Complete IV and VIF report for 104 features
├── compute_iv_vif.py                # Standalone Information Value & VIF calculator
├── requirements.txt                 # Python dependencies for cloud deployment
└── README.md                        # Project documentation
```

---

## 💻 Local Setup & Execution
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/banking-credit-risk-ai.git
   cd banking-credit-risk-ai
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit portal:
   ```bash
   streamlit run app.py
   ```
