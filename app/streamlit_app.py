"""
Streamlit deployment app for the Sydney Housing Price Prediction model.
Run with: streamlit run streamlit_app.py
"""
import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Sydney Housing Price Predictor", page_icon="🏠", layout="centered")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "model.joblib"))
meta = joblib.load(os.path.join(BASE_DIR, "model_meta.joblib"))

st.title("🏠 Sydney Housing Price Predictor")
st.write(
    f"Predicts sale price for houses, townhouses and villas in "
    f"**{', '.join(meta['suburbs'])}**, trained using a {meta['best_model_name']} model."
)

with st.form("property_form"):
    col1, col2 = st.columns(2)
    with col1:
        suburb = st.selectbox("Suburb", meta['suburbs'])
        property_type = st.selectbox("Property type", meta['property_types'])
        bedrooms = st.number_input("Bedrooms", 0, 10, 3)
        bathrooms = st.number_input("Bathrooms", 0, 10, 2)
        car_spaces = st.number_input("Car spaces", 0, 10, 1)
        storeys = st.number_input("Storeys", 1, 5, 2)
    with col2:
        land_size = st.number_input("Land size (m²)", 0.0, 5000.0, 500.0, step=10.0)
        building_size = st.number_input("Building size (m²)", 0.0, 2000.0, 200.0, step=10.0)
        year_built = st.number_input("Year built", 1900, 2026, 2000)
        pool = st.checkbox("Has pool?")
        renovated = st.checkbox("Recently renovated?")

    description = st.text_area(
        "Listing description (optional — used to count words / 'luxury' keywords)",
        placeholder="e.g. Stunning family home with spectacular views..."
    )

    submitted = st.form_submit_button("Predict sale price")

if submitted:
    property_age = max(0, 2026 - year_built)
    desc = description or ""
    luxury_words = ['luxury', 'stunning', 'exceptional', 'rare', 'premier', 'elegant',
                     'exquisite', 'spectacular', 'grand', 'prestigious', 'resort']
    luxury_count = sum(w in desc.lower() for w in luxury_words)

    row = pd.DataFrame([{
        'bedrooms': bedrooms, 'bathrooms': bathrooms, 'car_spaces': car_spaces,
        'land_size_m2': land_size, 'building_size_m2': building_size,
        'property_age': property_age, 'Storeys': storeys,
        'Pool': int(pool), 'renovated': int(renovated),
        'desc_word_count': len(desc.split()), 'luxury_word_count': luxury_count,
        'suburb': suburb, 'property_type': property_type
    }])

    pred = model.predict(row)[0]
    st.success(f"### Estimated sale price: ${pred:,.0f}")
    st.caption(
        "This is a prototype estimate from a small training dataset (135 sold properties "
        "across 3 suburbs) and should not be used as a substitute for professional valuation."
    )
