import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ------------------------
# Load Model & Dataset
# ------------------------
model = pickle.load(open("model.pkl", "rb"))
data = pd.read_csv("Cleaned_data.csv")   # for unique locations

# ------------------------
# Streamlit Config
# ------------------------
st.set_page_config(page_title="Pune House Price Prediction", page_icon="🏠", layout="centered")

# 🔹 Animated Background CSS
page_bg = """
<style>
@keyframes gradient {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}
.stApp {
    background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
    background-size: 400% 400%;
    animation: gradient 15s ease infinite;
}
div.stButton > button {
    background-color: #ff6b6b;
    color: white;
    border-radius: 10px;
    padding: 0.6em 1.2em;
    border: none;
    font-weight: bold;
}
div.stButton > button:hover {
    background-color: #e73c7e;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ------------------------
# UI
# ------------------------
st.title("🏠 Pune House Price Prediction")

# Inputs
sqft = st.number_input("Total Area (sqft)", min_value=300, max_value=10000, value=1000, step=50)
bath = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2, step=1)
bhk = st.number_input("Number of BHK", min_value=1, max_value=10, value=2, step=1)
balcony = st.number_input("Number of Balconies", min_value=0, max_value=5, value=1, step=1)

# Locations from dataset
locations = sorted(data['location'].unique())
location = st.selectbox("Select Location", locations)

# ------------------------
# Prediction Function
# ------------------------
def predict_price(location, sqft, bath, bhk, balcony):
    input_data = pd.DataFrame([[location, sqft, bath, bhk, balcony]],
                              columns=['location','total_sqft','bath','bhk','balcony'])
    return round(model.predict(input_data)[0], 2)

# ------------------------
# Predict Button
# ------------------------
if st.button("Predict Price"):
    result = predict_price(location, sqft, bath, bhk, balcony)
    st.success(f"🏡 Predicted Price: ₹ {result} Lakhs")
