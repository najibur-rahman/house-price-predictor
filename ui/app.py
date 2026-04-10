import streamlit as st
import requests
import json

import os
API_URL = os.getenv("API_URL", "https://house-price-predictor-dd0b.onrender.com")

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 California House Price Predictor")
st.markdown("Predict California house prices using XGBoost model")
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📍 Location")
    longitude = st.slider("Longitude", -124.0, -114.0, -118.49)
    latitude = st.slider("Latitude", 32.0, 42.0, 34.26)
    ocean = st.selectbox("Ocean Proximity", [
        "NEAR BAY", "INLAND", "NEAR OCEAN", "ISLAND", "<1H OCEAN"
    ])

with col2:
    st.subheader("🏘️ House Info")
    housing_median_age = st.slider("House Age (years)", 1, 52, 29)
    total_rooms = st.number_input("Total Rooms", 100, 10000, 2127)
    total_bedrooms = st.number_input("Total Bedrooms", 50, 3000, 435)

with col3:
    st.subheader("👥 Neighborhood")
    population = st.number_input("Population", 100, 10000, 1166)
    households = st.number_input("Households", 50, 3000, 409)
    median_income = st.slider("Median Income ($10k)", 0.5, 15.0, 3.53)

ocean_proximity_INLAND = 1 if ocean == "INLAND" else 0
ocean_proximity_ISLAND = 1 if ocean == "ISLAND" else 0
ocean_proximity_NEAR_BAY = 1 if ocean == "NEAR BAY" else 0
ocean_proximity_NEAR_OCEAN = 1 if ocean == "NEAR OCEAN" else 0

rooms_per_household = round(total_rooms / max(households, 1), 2)
bedrooms_per_room = round(total_bedrooms / max(total_rooms, 1), 2)
population_per_household = round(population / max(households, 1), 2)

if median_income <= 1.5:
    income_bracket = 1.0
elif median_income <= 3.0:
    income_bracket = 2.0
elif median_income <= 4.5:
    income_bracket = 3.0
elif median_income <= 6.0:
    income_bracket = 4.0
else:
    income_bracket = 5.0

st.divider()

col_info1, col_info2, col_info3 = st.columns(3)
with col_info1:
    st.metric("Rooms per Household", rooms_per_household)
with col_info2:
    st.metric("Bedrooms per Room", bedrooms_per_room)
with col_info3:
    st.metric("Population per Household", population_per_household)

st.divider()

if st.button("🔮 Predict House Price", type="primary", use_container_width=True):
    payload = {
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": float(housing_median_age),
        "total_rooms": float(total_rooms),
        "total_bedrooms": float(total_bedrooms),
        "population": float(population),
        "households": float(households),
        "median_income": median_income,
        "ocean_proximity_INLAND": ocean_proximity_INLAND,
        "ocean_proximity_ISLAND": ocean_proximity_ISLAND,
        "ocean_proximity_NEAR_BAY": ocean_proximity_NEAR_BAY,
        "ocean_proximity_NEAR_OCEAN": ocean_proximity_NEAR_OCEAN,
        "rooms_per_household": rooms_per_household,
        "bedrooms_per_room": bedrooms_per_room,
        "population_per_household": population_per_household,
        "income_bracket": income_bracket
    }

    try:
        with st.spinner("Predicting..."):
            response = requests.post(f"{API_URL}/predict", json=payload)

        if response.status_code == 200:
            result = response.json()
            st.success("Prediction Complete!")

            res_col1, res_col2, res_col3 = st.columns(3)
            with res_col1:
                st.metric("Predicted Price", result['predicted_price_formatted'])
            with res_col2:
                st.metric("Model Used", result['model_used'])
            with res_col3:
                monthly = result['predicted_price'] / 240
                st.metric("Est. Monthly Payment", f"${monthly:,.0f}")

            st.balloons()

        else:
            st.error(f"API Error: {response.status_code}")

    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Please make sure FastAPI is running.")
    except Exception as e:
        st.error(f"Error: {str(e)}")

st.divider()
st.caption("House Price Predictor | XGBoost Model | California Housing Dataset")