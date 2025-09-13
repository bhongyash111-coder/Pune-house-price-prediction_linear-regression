import pandas as pd
import pickle as pkl
import streamlit as st

# Load model and data
model = pkl.load(open("model.pkl", "rb"))
data = pd.read_csv("cleaned_data.csv")

# 🔹 Inject custom CSS for animated background
page_bg = """
<style>
/* Full page background */
.stApp {
    background: linear-gradient((-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
    background-size: 800% 800%;
    animation: gradientShift 20s ease infinite;
}

/* Keyframes for smooth transition */
@keyframes gradientShift {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# 🔹 App UI
st.header("Pune House Price Prediction")

loc = st.selectbox("Select Location", sorted(data['site_location'].unique()))
total_sqft = st.number_input("Enter Total Square Footage", min_value=300.0, value=1000.0, step=50.0)
bhk = st.number_input("Select Number of Bedrooms", min_value=1, value=2, step=1)
bath = st.number_input("Select Number of Bathrooms", min_value=1, value=2, step=1)
balcony = st.number_input("Select Number of Balconies", min_value=0, value=1, step=1)

input_data = pd.DataFrame([[loc, bhk, total_sqft, bath, balcony]],
                          columns=['site_location', 'bedrooms', 'total_sqft', 'bath', 'balcony'])

if st.button("Predict"):
    output = model.predict(input_data)
    output_str = f"predicted rate is ₹ " + f"{output[0]:.2f}"
    # output_str = f"predicted rate is ₹ {str{output[0]:.2f}"
    st.success(output_str)
