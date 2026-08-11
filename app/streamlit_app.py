"""
Streamlit deployment app for the Sydney Housing Price Prediction model.

Run locally:
streamlit run app/streamlit_app.py
"""

import os
import streamlit as st
import pandas as pd
import joblib


# Page configuration
st.set_page_config(
    page_title="Sydney Housing Price Predictor",
    page_icon="🏠",
    layout="centered"
)


# Model path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# Load model safely
@st.cache_resource
def load_model():
    model_path = os.path.join(BASE_DIR, "model.joblib")
    meta_path = os.path.join(BASE_DIR, "model_meta.joblib")

    model = joblib.load(model_path)
    meta = joblib.load(meta_path)

    return model, meta


model, meta = load_model()


# App title
st.title("🏠 Sydney Housing Price Predictor")

st.write(
    f"Predicts sale price for houses, townhouses and villas in "
    f"**{', '.join(meta['suburbs'])}** "
    f"using a **{meta['best_model_name']}** model."
)


# Input form
with st.form("property_form"):

    col1, col2 = st.columns(2)

    with col1:

        suburb = st.selectbox(
            "Suburb",
            meta["suburbs"]
        )

        property_type = st.selectbox(
            "Property type",
            meta["property_types"]
        )

        bedrooms = st.number_input(
            "Bedrooms",
            min_value=0,
            max_value=10,
            value=3
        )

        bathrooms = st.number_input(
            "Bathrooms",
            min_value=0,
            max_value=10,
            value=2
        )

        car_spaces = st.number_input(
            "Car spaces",
            min_value=0,
            max_value=10,
            value=1
        )

        storeys = st.number_input(
            "Storeys",
            min_value=1,
            max_value=5,
            value=2
        )


    with col2:

        land_size = st.number_input(
            "Land size (m²)",
            min_value=0.0,
            max_value=5000.0,
            value=500.0,
            step=10.0
        )

        building_size = st.number_input(
            "Building size (m²)",
            min_value=0.0,
            max_value=2000.0,
            value=200.0,
            step=10.0
        )

        year_built = st.number_input(
            "Year built",
            min_value=1900,
            max_value=2026,
            value=2000
        )

        pool = st.checkbox(
            "Has pool?"
        )

        renovated = st.checkbox(
            "Recently renovated?"
        )


    description = st.text_area(
        "Listing description (optional — used to count words / luxury keywords)",
        placeholder="Example: Stunning family home with spectacular views..."
    )


    submitted = st.form_submit_button(
        "Predict sale price"
    )


# Prediction
if submitted:

    property_age = max(
        0,
        2026 - year_built
    )


    desc = description or ""


    luxury_words = [
        "luxury",
        "stunning",
        "exceptional",
        "rare",
        "premier",
        "elegant",
        "exquisite",
        "spectacular",
        "grand",
        "prestigious",
        "resort"
    ]


    luxury_count = sum(
        word in desc.lower()
        for word in luxury_words
    )


    input_data = pd.DataFrame([{

        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "car_spaces": car_spaces,
        "land_size_m2": land_size,
        "building_size_m2": building_size,
        "property_age": property_age,
        "Storeys": storeys,
        "Pool": int(pool),
        "renovated": int(renovated),
        "desc_word_count": len(desc.split()),
        "luxury_word_count": luxury_count,
        "suburb": suburb,
        "property_type": property_type

    }])


    prediction = model.predict(input_data)[0]


    st.success(
        f"### Estimated sale price: ${prediction:,.0f}"
    )


    st.caption(
        "This is a prototype estimate trained on a small dataset "
        "(135 sold properties across 3 suburbs). "
        "It should not replace a professional property valuation."
    )