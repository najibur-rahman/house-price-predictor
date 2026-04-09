from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pickle
import numpy as np
import os

app = FastAPI(
    title="House Price Predictor API",
    description="California house price prediction using XGBoost",
    version="1.0.0"
)

MODEL_PATH = "models/best_model.pkl"
SCALER_PATH = "models/scaler.pkl"

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

class HouseFeatures(BaseModel):
    longitude: float = Field(default=-118.49, description="Longitude")
    latitude: float = Field(default=34.26, description="Latitude")
    housing_median_age: float = Field(default=29.0, description="Median age of houses")
    total_rooms: float = Field(default=2127.0, description="Total rooms")
    total_bedrooms: float = Field(default=435.0, description="Total bedrooms")
    population: float = Field(default=1166.0, description="Population")
    households: float = Field(default=409.0, description="Number of households")
    median_income: float = Field(default=3.53, description="Median income (in $10,000s)")
    ocean_proximity_INLAND: int = Field(default=0, description="1 if inland")
    ocean_proximity_ISLAND: int = Field(default=0, description="1 if island")
    ocean_proximity_NEAR_BAY: int = Field(default=0, description="1 if near bay")
    ocean_proximity_NEAR_OCEAN: int = Field(default=0, description="1 if near ocean")
    rooms_per_household: float = Field(default=5.2, description="Rooms per household")
    bedrooms_per_room: float = Field(default=0.2, description="Bedrooms per room")
    population_per_household: float = Field(default=2.85, description="Population per household")
    income_bracket: float = Field(default=3.0, description="Income bracket (1-5)")

class PredictionResponse(BaseModel):
    predicted_price: float
    predicted_price_formatted: str
    model_used: str

@app.get("/")
def root():
    return {
        "message": "House Price Predictor API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_price(features: HouseFeatures):
    try:
        input_data = np.array([[
            features.longitude,
            features.latitude,
            features.housing_median_age,
            features.total_rooms,
            features.total_bedrooms,
            features.population,
            features.households,
            features.median_income,
            features.ocean_proximity_INLAND,
            features.ocean_proximity_ISLAND,
            features.ocean_proximity_NEAR_BAY,
            features.ocean_proximity_NEAR_OCEAN,
            features.rooms_per_household,
            features.bedrooms_per_room,
            features.population_per_household,
            features.income_bracket
        ]])

        prediction = model.predict(input_data)[0]
        prediction = max(0, float(prediction))

        return PredictionResponse(
            predicted_price=round(prediction, 2),
            predicted_price_formatted=f"${prediction:,.0f}",
            model_used="XGBoost"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sample-input")
def sample_input():
    return {
        "description": "Sample input for prediction",
        "sample": {
            "longitude": -118.49,
            "latitude": 34.26,
            "housing_median_age": 29.0,
            "total_rooms": 2127.0,
            "total_bedrooms": 435.0,
            "population": 1166.0,
            "households": 409.0,
            "median_income": 3.53,
            "ocean_proximity_INLAND": 0,
            "ocean_proximity_ISLAND": 0,
            "ocean_proximity_NEAR_BAY": 0,
            "ocean_proximity_NEAR_OCEAN": 0,
            "rooms_per_household": 5.2,
            "bedrooms_per_room": 0.2,
            "population_per_household": 2.85,
            "income_bracket": 3.0
        }
    }