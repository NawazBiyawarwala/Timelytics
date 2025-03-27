import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV

# Load data
df = pd.read_csv("data/delivery_data.csv")
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

# Feature engineering
df["distance"] = df["destination"].map(distance_matrix)
df["speed_efficiency"] = df["shipping_method"].map(
    {"Standard": 1, "Express": 1.2, "Overnight": 1.5}
)

# Features & target
X = df[
    [
        "product_category",
        "shipping_method",
        "distance",
        "order_quantity",
        "speed_efficiency",
    ]
]
y = df["delivery_time"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(), ["product_category", "shipping_method"]),
        ("num", StandardScaler(), ["distance", "order_quantity", "speed_efficiency"]),
    ],
    remainder="passthrough",
)

# Model pipeline
model = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("regressor", XGBRegressor(random_state=42, n_jobs=-1)),
    ]
)

# Hyperparameter tuning
param_grid = {
    "regressor__n_estimators": [500, 700],
    "regressor__learning_rate": [0.01, 0.05],
    "regressor__max_depth": [5, 7],
    "regressor__subsample": [0.8, 0.9],
}

grid_search = GridSearchCV(
    model, param_grid, cv=5, scoring="neg_mean_absolute_error", verbose=2
)
grid_search.fit(X_train, y_train)

# Save best model
best_model = grid_search.best_estimator_
joblib.dump(best_model, "delivery_time_model.pkl")

# Evaluate
print(f"Best MAE: {-grid_search.best_score_:.2f} hours")
print("Best Parameters:", grid_search.best_params_)
