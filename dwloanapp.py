import streamlit as st
import pickle
import pandas as pd
import numpy as np

# 1. Load the trained model, scaler, and features
with open('DW_model.pkl', 'rb') as file:
    assets = pickle.load(file)
    model = assets['model']
    scaler = assets['scaler']
    model_features = assets['features']

# 2. App Title and Styling
st.set_page_config(page_title='Loan Approval Predictor', layout='centered')
st.markdown("<h1 style='text-align: center; color: #2E86C1;'>Loan Approval Analysis Portal</h1>", unsafe_allow_html=True)

# 3. Input Form
st.header("Customer Financial Profile")
col1, col2 = st.columns(2)

with col1:
    requested_amt = st.number_input("Requested Loan Amount ($)", min_value=0, value=25000)
    fico = st.slider("FICO Score", min_value=300, max_value=850, value=650)
    income = st.number_input("Monthly Gross Income ($)", min_value=0, value=5000)

with col2:
    housing = st.number_input("Monthly Housing Payment ($)", min_value=0, value=1200)
    bankruptcy = st.selectbox("Ever Bankrupt or Foreclosed?", options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    lender = st.selectbox("Target Lender", options=['A', 'B', 'C'])

reason = st.selectbox("Reason for Loan", ['credit_card_refinancing', 'home_improvement', 'major_purchase', 'cover_an_unexpected_cost', 'debt_conslidation', 'other'])
employment_status = st.selectbox("Employment Status", ['full_time', 'part_time', 'unemployed'])
employment_sector = st.selectbox("Employment Sector", ['consumer_discretionary', 'information_technology', 'energy', 'finance', 'healthcare', 'industrials', 'utilities', 'Unknown'])

# 4. Prediction Logic
if st.button('Evaluate Application'):
    # Create raw DataFrame from inputs
    input_df = pd.DataFrame({
        'Requested_Loan_Amount': [requested_amt],
        'FICO_score': [fico],
        'Monthly_Gross_Income': [income],
        'Monthly_Housing_Payment': [housing],
        'Ever_Bankrupt_or_Foreclose': [bankruptcy],
        'Reason': [reason],
        'Employment_Status': [employment_status],
        'Employment_Sector': [employment_sector],
        'Lender': [lender]
    })

    # One-Hot Encoding
    input_encoded = pd.get_dummies(input_df)
    
    # Add missing columns with 0
    for col in model_features:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    
    # Reorder columns to match training exactly 
    input_final = input_encoded[model_features]

    # Scale the numerical inputs
    num_cols = ['Requested_Loan_Amount', 'FICO_score', 'Monthly_Gross_Income', 'Monthly_Housing_Payment']
    input_final[num_cols] = scaler.transform(input_final[num_cols])

    # 5. Predict
    prob = model.predict_proba(input_final)[:, 1][0]
    prediction = 1 if prob >= 0.3 else 0

    st.divider()
    if prediction == 1:
        st.success(f"✅ **Recommended for Approval** (Probability: {prob:.2f})")
        st.write("The applicant meets the risk threshold for the selected lender.")
    else:
        st.error(f"❌ **Recommendation: Deny** (Probability: {prob:.2f})")
        st.write("The applicant does not meet the criteria for this lender at the current settings.") 