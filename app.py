
# ============================================================
# LOAN DEFAULT PREDICTION APP
# ============================================================
#
# This application uses Streamlit to create a simple
# web interface for our Machine Learning model.
#
# The model was trained in Google Colab using:
# Gradient Boosting Classification
#
# Target variable:
#     defaulted
#
# 0 = No Default
# 1 = Default
# ============================================================


# ------------------------------------------------------------
# 1. IMPORT LIBRARIES
# ------------------------------------------------------------

import streamlit as st
import pandas as pd
import joblib


# ------------------------------------------------------------
# 2. LOAD THE SAVED MACHINE LEARNING MODEL
# ------------------------------------------------------------

# joblib.load() loads the model that we saved from Colab.
#
# The file "loan_default_model.pkl" must be in the
# same folder as this app.py file.

model = joblib.load("loan_default_model.pkl")


# ------------------------------------------------------------
# 3. PAGE CONFIGURATION/setting
# ------------------------------------------------------------

# This controls the appearance of the Streamlit page.

st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon=" ",
    layout="wide"
)


# ------------------------------------------------------------
# 4. APPLICATION TITLE
# ------------------------------------------------------------

st.title(" Loan Default Prediction System")

st.write(
    """
    This application uses a Machine Learning model to predict
    whether a borrower is likely to default on a loan.
    """
)

st.info(
    """
    Enter the borrower's information below and click
    **Predict Default**.
    """
)


# ============================================================
# 5. INPUT SECTION
# ============================================================
#
# The user enters information about a borrower.
#
# IMPORTANT:
# These inputs represent the variables used by the model.
# ============================================================

st.header(" Borrower Information")


# ------------------------------------------------------------
# CREATE TWO COLUMNS
# ------------------------------------------------------------
#
# Streamlit allows us to place inputs side by side.
#
# This makes the application easier to read.
# ------------------------------------------------------------

col1, col2 = st.columns(2)


# ============================================================
# COLUMN 1
# ============================================================

with col1:

    # Credit score
    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=850,
        value=650
    )

    # Loan amount / Exposure at Default
    ead = st.number_input(
        "Exposure at Default (EAD)",
        min_value=0.0,
        value=1000000.0
    )

    # Coupon rate
    coupon_rate = st.number_input(
        "Coupon Rate",
        min_value=0.0,
        value=5.0
    )

    # Leverage
    leverage = st.number_input(
        "Leverage",
        min_value=0.0,
        value=3.0
    )

    # Interest coverage
    interest_coverage = st.number_input(
        "Interest Coverage",
        min_value=0.0,
        value=2.0
    )

    # Debt-to-equity ratio
    debt_to_equity = st.number_input(
        "Debt-to-Equity",
        min_value=0.0,
        value=2.0
    )


# ============================================================
# COLUMN 2
# ============================================================

with col2:

    # Sector
    sector = st.selectbox(
        "Sector",
        [
            "Technology",
            "Healthcare",
            "Real_Estate",
            "Energy"
        ]
    )

    # Loan type
    loan_type = st.selectbox(
        "Loan Type",
        [
            "mortgage",
            "term_loan",
            "bond"
        ]
    )

    # Collateral
    collateral = st.selectbox(
        "Collateral",
        [
            "secured",
            "unsecured"
        ]
    )

    # Initial credit rating
    initial_rating = st.selectbox(
        "Initial Rating",
        [
            "A",
            "B",
            "BBB",
            "CCC"
        ]
    )

    # Maturity period
    maturity_months = st.number_input(
        "Maturity Months",
        min_value=1,
        value=36
    )

    # Annual probability of default
    pd_annual = st.number_input(
        "Annual Probability of Default",
        min_value=0.0,
        max_value=1.0,
        value=0.05
    )

    # Loss given default
    lgd = st.number_input(
        "Loss Given Default",
        min_value=0.0,
        max_value=1.0,
        value=0.50
    )


# ============================================================
# 6. PREDICTION BUTTON
# ============================================================

st.divider()

predict_button = st.button(
    "🔍 Predict Default",
    type="primary"
)


# ============================================================
# 7. MAKE THE PREDICTION
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # Create a DataFrame containing the user's information.
    # --------------------------------------------------------
    #
    # IMPORTANT:
    # The column names must match the feature names used
    # when the model was trained.
    # --------------------------------------------------------

    input_data = pd.DataFrame({

        "maturity_months": [maturity_months],

        "sector": [sector],

        "loan_type": [loan_type],

        "collateral": [collateral],

        "initial_rating": [initial_rating],

        "credit_score": [credit_score],

        "ead": [ead],

        "coupon_rate": [coupon_rate],

        "leverage": [leverage],

        "interest_coverage": [interest_coverage],

        "debt_to_equity": [debt_to_equity],

        "pd_annual": [pd_annual],

        "lgd": [lgd]

    })


    # --------------------------------------------------------
    # DISPLAY THE INPUT DATA/show information entered
    # --------------------------------------------------------

    st.subheader("Information Entered")

    st.dataframe(input_data)


    # --------------------------------------------------------
    # MAKE THE CLASS PREDICTION
    # --------------------------------------------------------
    #
    # predict() returns:
    #
    # 0 → No Default
    # 1 → Default
    # --------------------------------------------------------

    prediction = model.predict(input_data)[0]


    # --------------------------------------------------------
    # GET THE PROBABILITY OF DEFAULT
    # --------------------------------------------------------
    #
    # predict_proba() gives probabilities for the classes.
    #
    # Example:
    #
    # [0.80, 0.20]
    #
    # means:
    #
    # 80% probability of No Default
    # 20% probability of Default
    # --------------------------------------------------------

    probability = model.predict_proba(input_data)[0][1]


    # ========================================================
    # 8. DISPLAY THE RESULT
    # ========================================================

    st.subheader("Prediction Result")


    if prediction == 1:

        # If prediction is 1
        # the model predicts DEFAULT.

        st.error(" HIGH RISK: Predicted Default")

        st.write(
            f"Probability of Default: **{probability:.2%}**"
        )

    else:

        # If prediction is 0
        # the model predicts NO DEFAULT.

        st.success("LOW RISK: Predicted No Default")

        st.write(
            f"Probability of Default: **{probability:.2%}**"
        )


    # --------------------------------------------------------
    # 9. PROBABILITY BAR
    # --------------------------------------------------------

    st.subheader("Probability of Default")

    st.progress(float(probability))

    st.write(
        f"The model estimates a "
        f"**{probability:.2%}** probability of default."
    )



