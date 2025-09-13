# backend_fastapi.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import pickle
from typing import Optional

# ---- Load model and data ----
MODEL_PATH = "model.pkl"
DATA_PATH = "cleaned_data.csv"

try:
    model = pickle.load(open(MODEL_PATH, "rb"))
except Exception as e:
    raise RuntimeError(f"Could not load model from {MODEL_PATH}: {e}")

try:
    data = pd.read_csv(DATA_PATH)
    valid_locations = set(data['site_location'].dropna().unique())
except Exception:
    valid_locations = set()

# ---- FastAPI app ----
app = FastAPI(title="Pune House Price API")

# Allow your local frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Request schema ----
class PredictRequest(BaseModel):
    total_sqft: float = Field(..., gt=0)
    bedrooms: int = Field(..., ge=0)
    bath: int = Field(..., ge=0)
    balcony: int = Field(..., ge=0)
    site_location: str

# ---- Utility: build input DataFrame, compute sqft_per_bedroom if required ----
def build_input_df(req: PredictRequest):
    # canonical input
    row = {
        'total_sqft': float(req.total_sqft),
        'bath': int(req.bath),
        'balcony': int(req.balcony),
        'site_location': req.site_location,
        'bedrooms': int(req.bedrooms),
    }
    df = pd.DataFrame([row])

    # If model expects sqft_per_bedroom, compute it
    if hasattr(model, "feature_names_in_"):
        required = list(model.feature_names_in_)
        if 'sqft_per_bedroom' in required and 'sqft_per_bedroom' not in df.columns:
            # avoid division by zero
            df['sqft_per_bedroom'] = df['total_sqft'] / df['bedrooms'].replace(0, 1)
    return df

# ---- Prediction endpoint ----
@app.post("/predict")
def predict(req: PredictRequest):
    # sanitize location: if unknown, fallback to 'other' if present
    loc = req.site_location
    if valid_locations and loc not in valid_locations:
        if 'other' in valid_locations:
            loc = 'other'
        # else: keep original loc (pipeline's OneHotEncoder should use handle_unknown='ignore')

    req.site_location = loc
    input_df = build_input_df(req)

    # Try direct predict; if it fails, attempt to align dummies with feature_names_in_
    try:
        pred = model.predict(input_df)
    except Exception as e:
        # fallback: create dummies for site_location and align to feature names
        try:
            X = pd.get_dummies(input_df, columns=['site_location'], drop_first=True)
            if hasattr(model, "feature_names_in_"):
                for c in model.feature_names_in_:
                    if c not in X.columns:
                        X[c] = 0
                X = X[list(model.feature_names_in_)]
            pred = model.predict(X)
        except Exception as e2:
            raise HTTPException(status_code=500, detail=f"Prediction failed: {e}; fallback failed: {e2}")

    price_lakhs = float(pred[0])
    price_inr = int(round(price_lakhs * 100000))  # integer rupees

    return {"price_lakhs": price_lakhs, "price_inr": price_inr}
