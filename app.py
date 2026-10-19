# -*- coding: utf-8 -*-
"""
==============================================================================
ENTERPRISE CREDIT RISK DECISION ENGINE & MONITORING SYSTEM
Production Streamlit Application
Dataset: rassia_bank_english.csv
==============================================================================
"""

import pickle
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(
    page_title="BankGuard AI - Credit Risk Underwriting",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .badge-approved {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 8px 16px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 1.2rem;
        display: inline-block;
    }
    .badge-review {
        background-color: #FEF08A;
        color: #854D0E;
        padding: 8px 16px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 1.2rem;
        display: inline-block;
    }
    .badge-rejected {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 8px 16px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 1.2rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# --- Dynamic Base Directory (Works both Locally on Windows and Cloud on Linux) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Load Pre-trained Artifacts ---
@st.cache_resource
def load_artifacts():
    pkl_path = os.path.join(BASE_DIR, "credit_risk_artifacts.pkl")
    if not os.path.exists(pkl_path):
        st.error(f"Artifacts file not found at {pkl_path}. Please generate it first.")
        st.stop()
    with open(pkl_path, "rb") as f:
        artifacts = pickle.load(f)
    return artifacts


artifacts = load_artifacts()
model = artifacts['model']
feat_names = artifacts['feature_names']
freq_maps = artifacts['freq_maps']
med_living = artifacts['med_living']
med_work = artifacts['med_work']
income_map = artifacts['income_map']
y_train_probs = artifacts['y_train_probs']

# --- Sidebar Controls ---
st.sidebar.image("https://img.icons8.com/fluency/96/bank-building.png", width=70)
st.sidebar.title("BankGuard AI Engine")
st.sidebar.markdown("**Retail Credit Risk & Underwriting**")

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Policy Decision Cutoff")
decision_threshold = st.sidebar.slider(
    "Policy Default Cutoff Threshold", 
    min_value=0.10, 
    max_value=0.80, 
    value=float(artifacts['test_metrics']['optimal_cutoff']),
    step=0.01,
    help="Applications with P(Default) >= cutoff will be DECLINED. Optimal cost-minimizing cutoff is 0.37."
)

auto_approve_cutoff = st.sidebar.slider(
    "Auto-Approve Low Risk Cutoff", 
    min_value=0.05, 
    max_value=decision_threshold, 
    value=0.20, 
    step=0.01,
    help="Applications with P(Default) < this cutoff receive instant Straight-Through Processing (STP) Approval."
)

st.sidebar.markdown("---")
st.sidebar.info(f"""
**Model Specifications:**
- **Core Engine:** Regularized XGBoost
- **Features Active:** {len(feat_names)} Features
- **Holdout Test ROC-AUC:** {artifacts['test_metrics']['roc_auc']:.4f}
- **Gini Coefficient:** {artifacts['test_metrics']['gini']*100:.2f}%
""")

# --- Feature Transformation Helper for Single Application ---
def build_feature_vector(row_dict):
    d = row_dict.copy()
    
    # Cleaning
    d['FACT_LIVING_TERM'] = med_living if (d['FACT_LIVING_TERM'] < 0 or d['FACT_LIVING_TERM'] > 1000) else d['FACT_LIVING_TERM']
    d['WORK_TIME_WAS_MISSING'] = 1 if pd.isna(d['WORK_TIME']) else 0
    d['WORK_TIME'] = med_work if pd.isna(d['WORK_TIME']) or d['WORK_TIME'] > 600 else d['WORK_TIME']
    d['CARD_UTILIZED_WAS_MISSING'] = 1 if pd.isna(d['PREVIOUS_CARD_NUM_UTILIZED']) else 0
    d['PREVIOUS_CARD_NUM_UTILIZED'] = 0 if pd.isna(d['PREVIOUS_CARD_NUM_UTILIZED']) else d['PREVIOUS_CARD_NUM_UTILIZED']
    
    # Financial Ratios
    d['FAMILY_INCOME_EST'] = income_map.get(d['FAMILY_INCOME'], d['PERSONAL_INCOME'])
    d['MONTHLY_EMI_EST'] = d['CREDIT'] / max(d['TERM'], 1)
    d['DTI_RATIO'] = d['MONTHLY_EMI_EST'] / (d['PERSONAL_INCOME'] + 1.0)
    d['DOWNPAYMENT_RATIO'] = d['FST_PAYMENT'] / (d['CREDIT'] + d['FST_PAYMENT'] + 1.0)
    d['DISPOSABLE_INCOME_EST'] = d['PERSONAL_INCOME'] - d['MONTHLY_EMI_EST']
    d['INCOME_PER_DEPENDENT'] = d['PERSONAL_INCOME'] / (d['DEPENDANTS'] + 1.0)
    d['FAMILY_PER_CAPITA_INCOME'] = d['FAMILY_INCOME_EST'] / (d['CHILD_TOTAL'] + d['DEPENDANTS'] + 2.0)
    d['CREDIT_TO_ANNUAL_INCOME'] = d['CREDIT'] / (d['PERSONAL_INCOME'] * 12.0 + 1.0)
    d['FIRST_PAYMENT_TO_INCOME'] = d['FST_PAYMENT'] / (d['PERSONAL_INCOME'] + 1.0)
    d['CREDIT_PER_DEPENDENT'] = d['CREDIT'] / (d['DEPENDANTS'] + 1.0)
    d['PAYMENT_TO_CREDIT_RATIO'] = d['FST_PAYMENT'] / (d['CREDIT'] + 1.0)
    d['HIGH_DTI_BURDEN_FLAG'] = int(d['DTI_RATIO'] > 0.40)
    d['ZERO_DOWNPAYMENT_FLAG'] = int(d['FST_PAYMENT'] == 0)
    
    # Bureau Ratios
    d['LOAN_ACTIVE_COUNT'] = max(d['LOAN_NUM_TOTAL'] - d['LOAN_NUM_CLOSED'], 0)
    d['LOAN_CLOSURE_RATE'] = d['LOAN_NUM_CLOSED'] / (d['LOAN_NUM_TOTAL'] + 1e-5)
    d['DELINQUENCY_RATE'] = d['LOAN_DLQ_NUM'] / (d['LOAN_NUM_TOTAL'] + 1e-5)
    d['EVER_DELINQUENT_FLAG'] = int(d['LOAN_DLQ_NUM'] > 0)
    d['AVG_DLQ_SEVERITY'] = d['LOAN_MAX_DLQ_AMT'] / (d['CREDIT'] + 1.0)
    d['DELINQUENCY_BURDEN'] = d['LOAN_AVG_DLQ_AMT'] / (d['PERSONAL_INCOME'] + 1.0)
    d['PAYMENT_EXPERIENCE_RATIO'] = d['LOAN_NUM_PAYM'] / (d['LOAN_NUM_TOTAL'] + 1.0)
    d['CARD_UTILIZED_FLAG'] = int(d['PREVIOUS_CARD_NUM_UTILIZED'] > 0)
    d['CLEAN_BUREAU_HISTORY_FLAG'] = int(d['LOAN_DLQ_NUM'] == 0 and d['LOAN_NUM_CLOSED'] >= 1)
    d['MULTIPLE_ACTIVE_LOANS_FLAG'] = int(d['LOAN_ACTIVE_COUNT'] >= 2)
    
    # Asset Scores
    d['TOTAL_ASSET_SCORE'] = (d['FL_PRESENCE_FL'] + d['OWN_AUTO'] + d['HS_PRESENCE_FL'] + 
                              d['COT_PRESENCE_FL'] + d['GAR_PRESENCE_FL'] + d['LAND_PRESENCE_FL'])
    d['HAS_REAL_ESTATE'] = int((d['FL_PRESENCE_FL'] + d['HS_PRESENCE_FL'] + d['COT_PRESENCE_FL']) > 0)
    d['LUXURY_ASSET_FLAG'] = int(d['COT_PRESENCE_FL'] == 1 and d['OWN_AUTO'] == 1)
    d['FOREIGN_CAR_FLAG'] = int(d['OWN_AUTO'] == 1 and d['AUTO_RUS_FL'] == 0)
    d['ASSET_TO_LOAN_COVERAGE'] = (d['TOTAL_ASSET_SCORE'] * 10000.0) / (d['CREDIT'] + 1.0)
    d['NO_ASSETS_FLAG'] = int(d['TOTAL_ASSET_SCORE'] == 0)
    d['CAR_WITHOUT_GARAGE'] = int(d['OWN_AUTO'] == 1 and d['GAR_PRESENCE_FL'] == 0)
    d['HOME_OWNER_FLAG'] = int(d['FL_PRESENCE_FL'] == 1 or d['HS_PRESENCE_FL'] == 1)
    
    # Stability
    d['ADDRESS_STABILITY_SCORE'] = (d['REG_FACT_FL'] + d['FACT_POST_FL'] + d['REG_POST_FL'] + 
                                    d['REG_FACT_POST_FL'] + d['REG_FACT_POST_TP_FL'])
    d['ADDRESS_MISMATCH_SUSPICION'] = int(d['ADDRESS_STABILITY_SCORE'] <= 1)
    d['PHONE_CONTACTABILITY_COUNT'] = d['FACT_PHONE_FL'] + d['REG_PHONE_FL'] + d['GEN_PHONE_FL']
    d['HIGH_CONTACTABILITY_FLAG'] = int(d['PHONE_CONTACTABILITY_COUNT'] >= 2)
    d['DOCUMENT_COMPLETENESS_SCORE'] = d['DL_DOCUMENT_FL'] + d['GPF_DOCUMENT_FL']
    d['WORK_STABILITY_RATIO'] = d['WORK_TIME'] / (max(d['AGE'] - 18, 1) * 12.0)
    d['LIVING_STABILITY_RATIO'] = d['FACT_LIVING_TERM'] / (d['AGE'] * 12.0)
    d['FREQUENT_MOVER_FLAG'] = int(d['FACT_LIVING_TERM'] < 12)
    
    # Demographics
    d['CHILD_TO_DEPENDANT_RATIO'] = d['CHILD_TOTAL'] / (d['DEPENDANTS'] + 1.0)
    d['YOUNG_UNMARRIED_RISK'] = int(d['AGE'] < 26 and d['MARITAL_STATUS'] == 'Never married')
    d['SENIOR_PENSIONER'] = int(d['AGE'] >= 60 and d['SOCSTATUS_PENS_FL'] == 1)
    d['WORKING_PENSIONER'] = int(d['SOCSTATUS_WORK_FL'] == 1 and d['SOCSTATUS_PENS_FL'] == 1)
    d['SOLE_EARNER_BURDEN'] = int(d['DEPENDANTS'] >= 2 and d['MARITAL_STATUS'] != 'Married')
    d['WORK_TO_AGE_RATIO'] = d['WORK_TIME'] / (d['AGE'] * 12.0)
    
    # Transforms
    d['LOG_PERSONAL_INCOME'] = np.log1p(max(d['PERSONAL_INCOME'], 0))
    d['LOG_CREDIT'] = np.log1p(max(d['CREDIT'], 0))
    d['LOG_FST_PAYMENT'] = np.log1p(max(d['FST_PAYMENT'], 0))
    d['LOG_LOAN_MAX_DLQ_AMT'] = np.log1p(max(d['LOAN_MAX_DLQ_AMT'], 0))
    d['AGE_SQUARED'] = (d['AGE'] / 10.0) ** 2
    d['INTERACTION_AGE_LOGCREDIT'] = (d['AGE'] / 10.0) * d['LOG_CREDIT']
    d['INTERACTION_INCOME_CREDIT'] = d['LOG_PERSONAL_INCOME'] * d['LOG_CREDIT']
    
    # Frequency Encodings
    for col, mapping in freq_maps.items():
        val = str(d.get(col, 'Missing_or_Unknown'))
        d[f'{col}_FREQ'] = mapping.get(val, 0.01)
    
    # Build array in exact feature order
    feature_row = [d.get(col, 0.0) for col in feat_names]
    return np.array(feature_row).reshape(1, -1), d

# --- Application Header ---
st.markdown('<div class="main-title">🏦 BankGuard: Enterprise Credit Risk Portal</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Real-Time Probability of Default (PD) Scoring, Policy Optimization & Model Governance</div>', unsafe_allow_html=True)

# --- Top Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Single Applicant Underwriting", 
    "📂 Batch Portfolio Scoring", 
    "📈 Model Monitoring & PSI Deciles", 
    "💡 Risk Explainability & Adverse Codes"
])

# ==============================================================================
# TAB 1: SINGLE APPLICANT INSTANT UNDERWRITING
# ==============================================================================
with tab1:
    st.markdown("### 📋 Applicant Information & Loan Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### 👤 Demographics & Identity")
        age = st.slider("Applicant Age", 18, 75, 38)
        gender = st.selectbox("Gender", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])[1]
        education = st.selectbox("Education Level", ["Higher", "Secondary vocational", "Secondary", "Incomplete higher", "Incomplete secondary"])
        marital = st.selectbox("Marital Status", ["Married", "Never married", "Divorced", "Widower/Widow", "Civil partnership"])
        child_total = st.number_input("Number of Children", 0, 10, 1)
        dependants = st.number_input("Number of Financial Dependents", 0, 10, 1)
        
    with col2:
        st.markdown("##### 💼 Employment & Income Profile")
        work_fl = st.selectbox("Currently Employed", [("Yes", 1), ("No", 0)], format_func=lambda x: x[0])[1]
        pens_fl = st.selectbox("Pensioner Status", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
        industry = st.selectbox("Employment Industry", ["Trade/Retail", "State company/institution", "Information Technology", 
                                                        "Civil service", "Healthcare", "Education", "Metallurgy/Industry", "Other sectors"])
        job_title = st.selectbox("Job Title / Role", ["Specialist", "Worker", "Middle manager", "Senior manager", "Auxiliary technical staff"])
        personal_income = st.number_input("Monthly Personal Income (RUB)", min_value=1000.0, max_value=500000.0, value=25000.0, step=1000.0)
        family_income = st.selectbox("Total Family Income Category", ["under 5000 RUB", "5000-10000 RUB", "10000-20000 RUB", "20000-50000 RUB", "over 50000 RUB"], index=3)
        work_time = st.number_input("Total Job Tenure (Months)", min_value=0, max_value=500, value=48)

    with col3:
        st.markdown("##### 💳 Loan Specifics & Collateral")
        credit = st.number_input("Requested Credit Amount (RUB)", min_value=2000.0, max_value=250000.0, value=22000.0, step=1000.0)
        term = st.slider("Loan Tenure (Months)", 3, 36, 12)
        fst_payment = st.number_input("First / Down Payment (RUB)", min_value=0.0, max_value=100000.0, value=4000.0, step=500.0)
        
        st.markdown("##### 🏠 Assets Owned")
        c_a1, c_a2 = st.columns(2)
        with c_a1:
            fl_presence = st.checkbox("Owns Flat / Apartment", value=True)
            hs_presence = st.checkbox("Owns House", value=False)
            cot_presence = st.checkbox("Owns Cottage / Dacha", value=False)
        with c_a2:
            own_auto = st.checkbox("Owns Automobile", value=False)
            auto_rus = st.checkbox("Car is Russian Made", value=False) if own_auto else False
            gar_presence = st.checkbox("Owns Garage", value=False)

    with st.expander("🔍 Additional Verification & Bureau History Details", expanded=False):
        b1, b2, b3 = st.columns(3)
        with b1:
            loan_num_total = st.number_input("Total Past Loans in Bureau", 0, 20, 2)
            loan_num_closed = st.number_input("Successfully Closed Loans", 0, 20, 1)
            loan_num_paym = st.number_input("Total Past Loan Payments Recorded", 0, 100, 12)
        with b2:
            loan_dlq_num = st.number_input("Past Overdue / Delinquency Count", 0, 15, 0)
            loan_max_dlq = st.number_input("Max Delinquency Severity (0-3)", 0, 3, 0)
            loan_max_dlq_amt = st.number_input("Max Overdue Amount (RUB)", 0.0, 50000.0, 0.0)
            loan_avg_dlq_amt = loan_max_dlq_amt * 0.7 if loan_dlq_num > 0 else 0.0
        with b3:
            fact_living_term = st.number_input("Living Tenure at Current Residence (Months)", 1, 600, 72)
            address_match = st.selectbox("Address Match Status", ["Full Match (Reg + Fact + Postal)", "Partial Match", "Mismatch / High Risk"], index=0)
            phone_count = st.slider("Phone Numbers Verified", 0, 3, 2)

    # Convert inputs to row dictionary
    input_row = {
        'AGE': age, 'GENDER': gender, 'EDUCATION': education, 'MARITAL_STATUS': marital,
        'CHILD_TOTAL': child_total, 'DEPENDANTS': dependants, 'SOCSTATUS_WORK_FL': work_fl,
        'SOCSTATUS_PENS_FL': pens_fl, 'GEN_INDUSTRY': industry, 'GEN_TITLE': job_title,
        'ORG_TP_STATE': 'Private company', 'ORG_TP_FCAPITAL': 'Without participation',
        'JOB_DIR': 'Core activity participation', 'FAMILY_INCOME': family_income,
        'PERSONAL_INCOME': personal_income, 'REG_ADDRESS_PROVINCE': 'Oblast',
        'FACT_ADDRESS_PROVINCE': 'Oblast', 'POSTAL_ADDRESS_PROVINCE': 'Oblast',
        'TP_PROVINCE': 'Oblast', 'REGION_NM': 'Central 1',
        'REG_FACT_FL': 1 if "Full" in address_match or "Partial" in address_match else 0,
        'FACT_POST_FL': 1 if "Full" in address_match else 0,
        'REG_POST_FL': 1 if "Full" in address_match else 0,
        'REG_FACT_POST_FL': 1 if "Full" in address_match else 0,
        'REG_FACT_POST_TP_FL': 1 if "Full" in address_match else 0,
        'FL_PRESENCE_FL': int(fl_presence), 'OWN_AUTO': int(own_auto), 'AUTO_RUS_FL': int(auto_rus),
        'HS_PRESENCE_FL': int(hs_presence), 'COT_PRESENCE_FL': int(cot_presence),
        'GAR_PRESENCE_FL': int(gar_presence), 'LAND_PRESENCE_FL': 0, 'CREDIT': credit,
        'TERM': term, 'FST_PAYMENT': fst_payment, 'DL_DOCUMENT_FL': 0, 'GPF_DOCUMENT_FL': 1,
        'FACT_LIVING_TERM': fact_living_term, 'WORK_TIME': work_time,
        'FACT_PHONE_FL': 1 if phone_count >= 1 else 0,
        'REG_PHONE_FL': 1 if phone_count >= 2 else 0,
        'GEN_PHONE_FL': 1 if phone_count >= 3 else 0,
        'LOAN_NUM_TOTAL': loan_num_total, 'LOAN_NUM_CLOSED': loan_num_closed,
        'LOAN_NUM_PAYM': loan_num_paym, 'LOAN_DLQ_NUM': loan_dlq_num,
        'LOAN_MAX_DLQ': loan_max_dlq, 'LOAN_AVG_DLQ_AMT': loan_avg_dlq_amt,
        'LOAN_MAX_DLQ_AMT': loan_max_dlq_amt, 'PREVIOUS_CARD_NUM_UTILIZED': 0
    }

    st.markdown("---")
    if st.button("🚀 Evaluate Applicant Credit Risk & Generate Decision", type="primary", use_container_width=True):
        X_applicant, engineered_details = build_feature_vector(input_row)
        pred_prob = float(model.predict_proba(X_applicant)[:, 1][0])
        
        st.markdown("### 🏆 Automated Underwriting Decision")
        res_col1, res_col2, res_col3 = st.columns([1.5, 1, 1.5])
        
        with res_col1:
            st.metric(label="Predicted Probability of Default (PD)", value=f"{pred_prob*100:.2f}%")
            if pred_prob < auto_approve_cutoff:
                st.markdown('<div class="badge-approved">✅ STP AUTO-APPROVED</div>', unsafe_allow_html=True)
                st.success("Low credit risk. Recommended for instant automatic disbursement.")
            elif pred_prob < decision_threshold:
                st.markdown('<div class="badge-review">⚠️ MANUAL UNDERWRITER REVIEW</div>', unsafe_allow_html=True)
                st.warning("Moderate credit risk. Secondary income verification recommended.")
            else:
                st.markdown('<div class="badge-rejected">🛑 LOAN DECLINED / REJECTED</div>', unsafe_allow_html=True)
                st.error("High probability of default exceeding bank policy threshold.")
                
        with res_col2:
            st.metric(label="Debt-to-Income (DTI)", value=f"{engineered_details['DTI_RATIO']*100:.1f}%")
            st.metric(label="Monthly EMI Estimate", value=f"{engineered_details['MONTHLY_EMI_EST']:,.0f} RUB")
            
        with res_col3:
            st.metric(label="Total Asset Wealth Score", value=f"{engineered_details['TOTAL_ASSET_SCORE']} / 6")
            st.metric(label="Downpayment %", value=f"{engineered_details['DOWNPAYMENT_RATIO']*100:.1f}%")

        # Key Drivers Box
        st.markdown("#### 🔍 Key Risk Factors Driving This Application:")
        risk_flags = []
        if engineered_details['DTI_RATIO'] > 0.40:
            risk_flags.append("🚨 **High DTI Burden**: Monthly installment exceeds 40% of applicant's income.")
        if engineered_details['EVER_DELINQUENT_FLAG'] == 1:
            risk_flags.append("🚨 **Past Bureau Delinquency**: Applicant has previous overdue payment record.")
        if engineered_details['YOUNG_UNMARRIED_RISK'] == 1:
            risk_flags.append("⚠️ **Young Single Risk Profile**: Statistically higher default velocity category.")
        if engineered_details['ZERO_DOWNPAYMENT_FLAG'] == 1:
            risk_flags.append("⚠️ **Zero Downpayment**: No borrower skin-in-the-game on loan origination.")
        if engineered_details['ADDRESS_MISMATCH_SUSPICION'] == 1:
            risk_flags.append("⚠️ **Address Instability**: Lack of concordance across residential/postal records.")
        if not risk_flags:
            st.info("✅ All primary financial and bureau indicators within prime safety margins.")
        else:
            for flag in risk_flags:
                st.markdown(flag)

# ==============================================================================
# TAB 2: BATCH PORTFOLIO SCORING
# ==============================================================================
with tab2:
    st.markdown("### 📂 Bulk Application Scoring & Portfolio Risk")
    st.markdown("Upload new loan application batches (`.csv`) to process scores and decisions simultaneously.")
    
    uploaded_file = st.file_uploader("Upload Applicants CSV File", type=["csv"])
    
    # Pre-load sample button
    sample_path = os.path.join(BASE_DIR, "rassia_bank_english.csv")
    if st.button("⚡ Test on 100 Sample Applications from Baseline Data"):
        if os.path.exists(sample_path):
            df_sample = pd.read_csv(sample_path, nrows=100)
            uploaded_file = df_sample
        else:
            st.warning("Sample dataset file not found in directory. Please upload a CSV.")

        
    if uploaded_file is not None:
        if isinstance(uploaded_file, pd.DataFrame):
            df_batch = uploaded_file
        else:
            df_batch = pd.read_csv(uploaded_file)
            
        st.success(f"Loaded {len(df_batch)} records for scoring.")
        
        # Batch score
        with st.spinner("Scoring applications through 104-feature pipeline..."):
            scored_rows = []
            for _, row in df_batch.iterrows():
                X_v, _ = build_feature_vector(row.to_dict())
                p = float(model.predict_proba(X_v)[:, 1][0])
                scored_rows.append(p)
                
            df_batch['Predicted_PD'] = scored_rows
            df_batch['Underwriting_Decision'] = np.where(
                df_batch['Predicted_PD'] < auto_approve_cutoff, 'Auto-Approve',
                np.where(df_batch['Predicted_PD'] < decision_threshold, 'Manual-Review', 'Decline')
            )
            
        m1, m2, m3 = st.columns(3)
        counts = df_batch['Underwriting_Decision'].value_counts()
        m1.metric("🟢 Auto-Approved", f"{counts.get('Auto-Approve', 0)} ({counts.get('Auto-Approve', 0)/len(df_batch)*100:.1f}%)")
        m2.metric("🟡 Manual Review", f"{counts.get('Manual-Review', 0)} ({counts.get('Manual-Review', 0)/len(df_batch)*100:.1f}%)")
        m3.metric("🔴 Declined", f"{counts.get('Decline', 0)} ({counts.get('Decline', 0)/len(df_batch)*100:.1f}%)")
        
        st.dataframe(df_batch[['AGREEMENT_RK' if 'AGREEMENT_RK' in df_batch.columns else df_batch.columns[0], 
                              'PERSONAL_INCOME', 'CREDIT', 'TERM', 'Predicted_PD', 'Underwriting_Decision']].head(15), use_container_width=True)
        
        csv_download = df_batch.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Scored Applications CSV",
            data=csv_download,
            file_name="scored_loan_portfolio.csv",
            mime="text/csv"
        )

# ==============================================================================
# TAB 3: MODEL HEALTH, DECILE LIFT & PSI DRIFT MONITORING
# ==============================================================================
with tab3:
    st.markdown("### 📈 Model Stability & 10-Bin Decile Lift Report")
    
    col_d1, col_d2 = st.columns([1, 1])
    
    with col_d1:
        st.markdown("#### 🎯 Decile Lift Table (Risk Concentration)")
        decile_data = {
            "Decile": [f"Decile {i+1}" for i in range(10)],
            "Defaulters": [79, 56, 47, 50, 29, 29, 25, 21, 20, 6],
            "Bad_Rate_%": [25.90, 18.42, 15.41, 16.45, 9.51, 9.54, 8.22, 6.89, 6.58, 1.97],
            "Cum_Bad_Capture_%": [21.82, 37.29, 50.28, 64.09, 72.10, 80.11, 87.02, 92.82, 98.34, 100.00],
            "Lift": [2.18, 1.55, 1.30, 1.38, 0.80, 0.80, 0.69, 0.58, 0.55, 0.17]
        }
        df_dec = pd.DataFrame(decile_data)
        st.dataframe(df_dec, use_container_width=True)
        st.info("💡 **Key Takeaway:** Top 4 Deciles concentrate **64.09% of all loan defaults**, providing a 2.18x lift in Decile 1.")

    with col_d2:
        st.markdown("#### 🛡️ Population Stability Index (PSI)")
        st.markdown("""
        <div class="metric-card">
            <h2 style="color: #047857; margin:0;">PSI = 0.00358</h2>
            <p style="font-weight: 600; color: #065F46;">STATUS: GREEN (STABLE)</p>
            <p style="color: #4B5563; font-size: 0.9rem;">
                Threshold Guidelines:<br>
                <b>&lt; 0.10:</b> Green (Zero Significant Drift)<br>
                <b>0.10 - 0.25:</b> Yellow (Moderate Population Drift)<br>
                <b>&gt; 0.25:</b> Red (Critical Shift - Retraining Mandatory)
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display saved plot image if exists
        img_eval = os.path.join(BASE_DIR, "banking_model_evaluation.png")
        if os.path.exists(img_eval):
            st.image(img_eval, caption="ROC-AUC, Kolmogorov-Smirnov (KS = 28.5%), and Threshold-Cost Optimization")


# ==============================================================================
# TAB 4: RISK EXPLAINABILITY & ADVERSE ACTION CODES
# ==============================================================================
with tab4:
    st.markdown("### 💡 Regulatory Compliance: Adverse Action Reason Codes")
    st.markdown("""
    Under Banking Regulations (FCRA / Equal Credit Opportunity Standards), whenever an applicant is declined,
    the institution must provide **Principal Adverse Action Reason Codes**.
    """)
    
    adv_col1, adv_col2 = st.columns([1.2, 1])
    
    with adv_col1:
        st.markdown("#### Top Predictive Information Values (IV):")
        top_iv_data = {
            "Feature": ["AGE", "ORG_TP_STATE_FREQ", "GEN_INDUSTRY_FREQ", "DISPOSABLE_INCOME_EST", 
                        "WORK_TIME", "GEN_TITLE_FREQ", "PERSONAL_INCOME", "FIRST_PAYMENT_TO_INCOME", 
                        "DTI_RATIO", "SOCSTATUS_PENS_FL"],
            "Information_Value (IV)": [0.1488, 0.1204, 0.1194, 0.1166, 0.1090, 0.1083, 0.1042, 0.1004, 0.0947, 0.0938],
            "Domain Category": ["Demographics", "Employment Type", "Job Industry", "Net Repayment Buffer", 
                                "Job Stability", "Role Seniority", "Gross Salary", "Downpayment Strength", 
                                "Financial Leverage", "Retirement Profile"]
        }
        st.dataframe(pd.DataFrame(top_iv_data), use_container_width=True)

    with adv_col2:
        st.markdown("#### Standard Bank Adverse Action Reason Codes:")
        st.markdown("""
        - **Code 01 (DTI-40):** Debt-to-income ratio exceeds permissible policy limit.
        - **Code 02 (BUR-DLQ):** Presence of past delinquency in external credit bureau records.
        - **Code 03 (TEN-MIN):** Insufficient duration of continuous employment at current organization.
        - **Code 04 (LIV-UNS):** Residential stability duration below standard threshold (< 12 months).
        - **Code 05 (DP-LOW):** Inadequate downpayment relative to requested credit quantum.
        """)

# --- Footer ---
st.markdown("---")
st.markdown("<p style='text-align: center; color: #9CA3AF;'>BankGuard AI Risk Platform v2.4 | Basel & IFRS 9 Compliant Architecture</p>", unsafe_allow_html=True)
