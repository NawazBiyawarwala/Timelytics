import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error

# Load artifacts
model = joblib.load("delivery_time_model.pkl")
test_df = pd.read_csv("data/test_data.csv")

# Preprocess test data
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

test_df["distance"] = test_df["destination"].map(distance_matrix)
test_df["speed_efficiency"] = test_df["shipping_method"].map(
    {"Standard": 1, "Express": 1.2, "Overnight": 1.5}
)

X_test = test_df[
    [
        "product_category",
        "shipping_method",
        "distance",
        "order_quantity",
        "speed_efficiency",
    ]
]
y_true = test_df["delivery_time"]

# Predict and evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_true, y_pred)

print(f"MAE: {mae:.2f} hours\n")
print("Sample Predictions vs Actual:")
print(pd.DataFrame({"Actual": y_true.head(), "Predicted": y_pred[:5].round(1)}))
