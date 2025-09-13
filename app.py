from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import pickle as pkl

# Load model and data
model = pkl.load(open("model.pkl", "rb"))
data = pd.read_csv("cleaned_data.csv")

app = FastAPI()

# ✅ Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # during dev, allow all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Pune House Price Prediction API is running 🚀"}

@app.post("/predict")
def predict(loc: str, sqft: float, bedrooms: int, bathrooms: int, balconies: int):
    # calculate derived feature
    sqft_per_bedroom = sqft / bedrooms if bedrooms > 0 else sqft

    # build input dataframe
    test_input = pd.DataFrame([[sqft, bathrooms, balconies, loc, sqft_per_bedroom]],
                              columns=['total_sqft', 'bath', 'balcony', 'site_location', 'sqft_per_bedroom'])

    # one-hot encode location
    test_input = pd.get_dummies(test_input, columns=['site_location'], drop_first=True)

    # align with training features
    for col in model.feature_names_in_:
        if col not in test_input:
            test_input[col] = 0
    test_input = test_input[model.feature_names_in_]

    # predict
    prediction = model.predict(test_input)[0]
    return {"predicted_price": round(float(prediction), 2)}
