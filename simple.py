import pandas as pd
import pickle as pkl
import streamlit as st

# Load model and data
model = pkl.load(open("model.pkl", "rb"))
data = pd.read_csv("cleaned_data.csv")

# 🔹 Bright and fast-changing background
page_bg = """
<style>
/* Full page background */
.stApp {
    background: linear-gradient(270deg, #ff6ec7, #ffb347, #ffff66, #50c878, #1e90ff, #da70d6);
    background-size: 1200% 1200%;
    animation: gradientShift 10s ease infinite;
}

/* Keyframes for smooth but fast transitions */
@keyframes gradientShift {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# 🔹 App UI
st.header("🏠 Pune House Price Prediction Linear Regression")

loc = st.selectbox("Select Location", sorted(data['site_location'].unique()))
total_sqft = st.number_input("Enter Total Square Footage", min_value=300.0, value=1000.0, step=50.0)
bhk = st.number_input("Select Number of Bedrooms", min_value=1, value=2, step=1)
bath = st.number_input("Select Number of Bathrooms", min_value=1, value=2, step=1)
balcony = st.number_input("Select Number of Balconies", min_value=0, value=1, step=1)

input_data = pd.DataFrame([[loc, bhk, total_sqft, bath, balcony]],
                          columns=['site_location', 'bedrooms', 'total_sqft', 'bath', 'balcony'])

if st.button("Predict"):
    output = model.predict(input_data)
    output_str = "predicted rate is ₹ " + str(int(output[0]))
    st.success(output_str)
