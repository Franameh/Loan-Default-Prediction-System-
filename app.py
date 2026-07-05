
import streamlit as st
import pandas as pd
import joblib

# PAGE CONFIGURATION

st.set_page_config(
    page_title='Loan Default Prediction',
    page_icon='💰',
    layout='wide',
    initial_sidebar_state='expanded'
)
# LOAD MODEL
model = joblib.load('loan_default_model.pkl')

# CUSTOM CSS

st.markdown('''
 <style>

 main {

 background-color: #FBFAC;
 }
 .title {
   text-align: center;
   color: #2c7fb8;
   font-size: 40px;
   font-weight: 700;
 }
 .subtitle {
   text-align: center;
   color: #647488;
   font-size: 18px;
   font-weight: 400;
 }
 .metric-card {
  background-color:white;
  padding:15px;
  border-radius:12px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.8);
 }

  .stButton > button {
   width:100%;
   height:55px;
   border-radius:12px;
   font-size:18px;
   font-weight:bold;
  }
  </style>
   ''', unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:

    st.image(
        "https://cdn.icons.png.flaticon.com/512/3135/3135715.png",
        width=150
    )

    st.title("[i] Project Overview")

    st.markdown('''
    ### Models used

    Logistic Regression
    Random Forest
    XGBoost

    ### Objective

   Predict whether a borrower is likely to default on a loan.

  ### Features
  Real-time prediction
  Risk probability
  ML-powered decision support
  ''')

    st.divider()

    st.info(
    "Built with Streamlit, Scikit-Learn and XGBOOST."
    )

# HEADERS

st.markdown(
    "<p class='title'>Loan Default Prediction System</p>",
    unsafe_allow_html=True
)

st.markdown(
    "<p class='subtitle'>Machine Learning Powered Credit Risk Assessment Dashboard</p>",
    unsafe_allow_html=True
)

st.divider()

# INPUT SECTION

col1, col2, col3 = st.columns(3)

with col1:

    Age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30,

    )

    Income = st.number_input(
        "Income",
        min_value=1000,
        value=50000,

    )
    LoanAmount = st.number_input(
        "Loan Amount",
        min_value=1000,
        value=10000,

    )

with col2:
    creditscore = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=850,
        value=650,
    )

    InterestRate = st.number_input(
        "Interest Rate(%)",
        min_value=1.0,
        max_value=40.0,
        value=12.5,

    )

    LoanTerm = st.number_input(
        "Loan Term(Months)",
        min_value=6,
        max_value=360,
        value=36,

    )
with col3:

    MonthsEmployed = st.number_input(
        "Months Employed",
        min_value=0,
        value=24,

    )

    NumCreditLines = st.number_input(
        "Number of Credit Lines",
        min_value=0,
        value=5,

    )

    DTIRatio = st.number_input(
        "Debt-to-Income Ratio",
        min_value=0.0,
        max_value=1.0,
        value=0.30,

    )

    #PREDICTION

    if st.button("Predict Loan Risk"):

        # Create a dictionary for input features
        input_dict = {
            'Age':[Age],
            'Income':[Income],
            'LoanAmount':[LoanAmount],
            'CreditScore':[creditscore],
            'MonthsEmployed':[MonthsEmployed],
            'NumCreditLines':[NumCreditLines],
            'DTIRatio':[DTIRatio],
            'InterestRate':[InterestRate],
            'LoanTerm':[LoanTerm],
        }

        # Get the column names from the training data X
        # This list is obtained from the kernel state's  'X' variable.
        expected_columns = [
            'Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
            'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio',
            'Education_High School', "Education_Master's",'Education_PhD',
            'Employment_Type_Pertime', 'Employment_Type_Self_emoloyed',
            'Employment_Type_Unemployed', 'MaritalStatus_Married', 'MaritalStatus_Single',
            'HasMortgage_Yes', 'HasDependent_Yes', 'LoanPurpose_Business', 'LoanPurpose_Education',
            'LoanPurpose_Home', 'LoanPurpose_Other', 'HasCosigner_Yes']

        # Create an empty DataFrame with one row of zeros and all expected columns
        final_input_df = pd.DataFrame(0, index=[0], columns=expected_columns)

        # Populate the relevant columns from the input_dict
        for key, value in input_dict.items():
            if key in final_input_df.columns:
                final_input_df[key] = value[0] # value is a list, take the first element

        prediction = model.predict(final_input_df)[0]

        probability = model.predict_proba(final_input_df)[0][1]

        st.divider()

        st.subheader("Prediction Result")

        colA, colB = st.columns(2)

        with colA:
            if prediction == 1:
                st.error(
                    f'''
                    # HIGH DEFAULT RISK
                    Probability of Default:
                    {probability:.2%}
                    '''
                )
            else:
                st.success(
                    f'''
                    # Low Default Risk
                    Confidence:
                    {(1-probability):.2%}
                    '''
                )

        with colB:
            st.metric(
                "Default Probability",
                f"{probability:.2%}"
            )

            st.progress(float(probability))
            st.metric(
                "Risk Level",
                "High Risk" if prediction == 1 else "Low Risk"
            )

# FOOTER
st.divider()
st.caption(
    "Loan Default Prediction Dashboard | Machine Learning Project"
)
