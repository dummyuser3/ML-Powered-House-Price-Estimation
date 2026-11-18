"""Streamlit interface for the existing house price prediction model."""

from pathlib import Path
import pickle

import joblib
import numpy as np
import streamlit as st

MODEL_PATH = Path(__file__).resolve().with_name("house_price_model.pkl")

st.set_page_config(
    page_title="House Price Prediction System",
    page_icon="🏠",
    layout="centered",
)

# Apply lightweight styling to give the form a clear, polished appearance.
st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(135deg, #f3f7ff 0%, #ffffff 55%, #eef8f5 100%);
        }
        .main .block-container {
            max-width: 900px;
            padding-top: 2.5rem;
        }
        div.stButton > button {
            width: 100%;
            min-height: 3rem;
            border-radius: 0.6rem;
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("House Price Prediction System")
st.markdown("### Machine Learning Powered House Price Estimation")
st.write(
    "Enter the property's details below to estimate its price using the "
    "existing Linear Regression model."
)

try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    model = None
    st.error(
        "The trained model file `house_price_model.pkl` was not found. "
        "Place the existing model file beside `streamlit_app.py`."
    )
except (OSError, EOFError, ImportError, ValueError, pickle.UnpicklingError) as error:
    model = None
    st.error(f"The trained model could not be loaded: {error}")

st.subheader("Property details")
area_column, bedroom_column, age_column = st.columns(3)

with area_column:
    area = st.number_input(
        "Area of House (square feet)",
        min_value=1.0,
        value=2000.0,
        step=100.0,
        help="Enter an area greater than zero.",
    )

with bedroom_column:
    bedrooms = st.number_input(
        "Number of Bedrooms",
        min_value=1,
        value=3,
        step=1,
        help="Enter a positive whole number.",
    )

with age_column:
    age = st.number_input(
        "Age of House (years)",
        min_value=0,
        value=5,
        step=1,
        help="Enter zero for a new house.",
    )

if st.button("Predict House Price", type="primary", disabled=model is None):
    # Validate and format the widget values for the model's three features.
    if area <= 0 or bedrooms <= 0 or age < 0:
        st.error("Enter a positive area, positive bedroom count, and non-negative age.")
    else:
        input_features = np.array([[area, bedrooms, age]], dtype=float)
        try:
            prediction = float(model.predict(input_features)[0])
        except (ValueError, TypeError, AttributeError, OSError, FloatingPointError) as error:
            st.error(f"Could not generate a prediction: {error}")
        else:
            if not np.isfinite(prediction):
                st.error("The model returned an invalid prediction. Check the input values.")
            else:
                st.success(f"Predicted House Price: ₹ {prediction:.2f} Lakhs")

st.caption("Estimates are produced by a sample model and are for demonstration only.")
