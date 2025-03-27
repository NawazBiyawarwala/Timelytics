import pandas as pd
import numpy as np
from faker import Faker

np.random.seed(42)
fake = Faker()

# Distance matrix from Mumbai (source) to key locations (in km)
distance_matrix = {
    "Mumbai": 0,  # Source
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


def generate_data(num_samples=10000):
    locations = list(distance_matrix.keys())
    base_speed = 50  # Average 50 km/h for logistics

    data = {
        "source_city": "Mumbai",  # Fixed source
        "destination": np.random.choice(locations, num_samples),
        "product_category": np.random.choice(
            ["Electronics", "Clothing", "Groceries", "Furniture"],
            num_samples,
            p=[0.2, 0.3, 0.3, 0.2],
        ),
        "shipping_method": np.random.choice(
            ["Standard", "Express", "Overnight"], num_samples, p=[0.6, 0.3, 0.1]
        ),
        "order_quantity": np.random.randint(1, 100, num_samples),
    }

    delivery_time = []
    for idx in range(num_samples):
        # Base time from distance
        distance = distance_matrix[data["destination"][idx]]
        base_time = (distance / base_speed) * 0.8  # 0.8 = efficiency factor

        # Product category adjustments
        if data["product_category"][idx] == "Electronics":
            base_time += 6
        elif data["product_category"][idx] == "Furniture":
            base_time += 18

        # Shipping method discounts
        if data["shipping_method"][idx] == "Express":
            base_time -= 0.2 * base_time
        elif data["shipping_method"][idx] == "Overnight":
            base_time -= 0.4 * base_time

        # Bulk order penalty (non-linear)
        if data["order_quantity"][idx] > 50:
            base_time += 24
        elif data["order_quantity"][idx] > 20:
            base_time += 12

        # Add noise
        delivery_time.append(base_time + np.random.randint(-3, 3))

    data["delivery_time"] = delivery_time
    df = pd.DataFrame(data)
    df.to_csv("data/delivery_data.csv", index=False)

    # Create test set
    test_df = df.sample(frac=0.2, random_state=42)
    test_df.to_csv("data/test_data.csv", index=False)


if __name__ == "__main__":
    generate_data()
