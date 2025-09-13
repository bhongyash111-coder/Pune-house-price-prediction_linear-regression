import os
import pickle as pkl
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

# To run this script:
# 1. Make sure you have the following installed:
#    pip install fastapi "uvicorn[standard]" pandas scikit-learn
# 2. Place this file in the same directory as your 'model.pkl' and 'cleaned_data.csv' files.
# 3. Open a terminal, navigate to this directory, and run:
#    uvicorn app:app --reload

# --- Initialize the FastAPI application ---
app = FastAPI(
    title="Pune House Price Prediction API",
    description="A simple API to predict house prices in Pune based on property features.",
    version="1.0.0",
)

# --- Enable CORS to allow requests from your HTML file ---
origins = [
    "http://127.0.0.1:8000",
    "http://localhost:8000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for simplicity. For production, specify your HTML file's host.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load the model and data (from your local files) ---
MODEL_PATH = "model.pkl"
DATA_PATH = "cleaned_data.csv"
model = None
data = None

try:
    with open(MODEL_PATH, "rb") as f:
        model = pkl.load(f)
    print("Model loaded successfully.")

    data = pd.read_csv(DATA_PATH)
    print("Cleaned data loaded successfully.")
    
except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure 'model.pkl' and 'cleaned_data.csv' are in the same directory.")
    print("Using a dummy model for demonstration purposes.")
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    dummy_df = pd.DataFrame({
        'total_sqft': [1000], 'bath': [2], 'balcony': [1], 'bedrooms': [2],
        'site_location_Aundh': [1], 'site_location_Balewadi': [0]
    })
    model.fit(dummy_df.drop('site_location_Aundh', axis=1), [50])
    model.feature_names_in_ = dummy_df.columns.drop('site_location_Aundh')
    data = dummy_df.copy()
except Exception as e:
    raise HTTPException(status_code=500, detail=f"An error occurred loading the files: {e}")

# --- Pydantic model for request body validation ---
class HouseFeatures(BaseModel):
    totalSqft: float
    bedrooms: int
    bathrooms: int
    balconies: int
    location: str

@app.post("/predict")
def predict(features: HouseFeatures):
    """
    Receives house features, performs a prediction using the loaded model,
    and returns the predicted price.
    """
    try:
        # Create a DataFrame for the model input
        input_data = pd.DataFrame([{
            'total_sqft': features.totalSqft,
            'bath': features.bathrooms,
            'balcony': features.balconies,
            'bedrooms': features.bedrooms,
        }])
        
        # Get all unique locations from the loaded dataset
        all_locations = sorted([str(loc) for loc in data['site_location'].dropna().unique()])

        # Create a new column for the selected location (one-hot encoding)
        location_column_name = f'site_location_{features.location}'
        input_data[location_column_name] = 1

        # Fill in other location columns with 0
        for loc in all_locations:
            if loc != features.location:
                input_data[f'site_location_{loc}'] = 0
                
        # Align columns with the model's feature names
        feature_names = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None
        if feature_names is not None:
            input_data = input_data.reindex(columns=feature_names, fill_value=0)

        # Make the prediction
        prediction = model.predict(input_data)
        price_lakhs = float(prediction[0])

        return {"price_lakhs": price_lakhs}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
