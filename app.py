import streamlit as st
import joblib
import pandas as pd

# Load model and distance matrix
model = joblib.load("delivery_time_model.pkl")
distance_matrix = {
    "Mumbai": 0,
    "Delhi": 1400,
    "Bangalore": 1000,
    "Kolkata": 2000,
    "Chennai": 1300,
    "Hyderabad": 700,
    "Pune": 150,
    "Nepal": 2500,
    "Bhutan": 2600,
    "Bangladesh": 2400,
    "Sri Lanka": 2700,
}

st.set_page_config(page_title="Timelytics", layout="wide")
st.title("Delivery Time Predictor")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        product_category = st.selectbox(
            "Product Category", ["Electronics", "Clothing", "Groceries", "Furniture"]
        )
        shipping_method = st.selectbox(
            "Shipping Method", ["Standard", "Express", "Overnight"]
        )

    with col2:
        destination = st.selectbox("Destination City", list(distance_matrix.keys()))
        order_quantity = st.number_input(
            "Order Quantity", min_value=1, max_value=100, value=1
        )

    submitted = st.form_submit_button("Predict Delivery Time")

if submitted:
    # Feature engineering
    input_data = pd.DataFrame(
        [
            [
                product_category,
                shipping_method,
                distance_matrix[destination],
                order_quantity,
                {"Standard": 1, "Express": 1.2, "Overnight": 1.5}[shipping_method],
            ]
        ],
        columns=[
            "product_category",
            "shipping_method",
            "distance",
            "order_quantity",
            "speed_efficiency",
        ],
    )

    prediction = model.predict(input_data)
    st.success(f"Estimated Delivery Time: {prediction[0]:.1f} hours")
