# Timelytics: Order-to-Delivery Time Prediction

A machine learning project to predict delivery times based on order details, deployed via Streamlit.

## Features

- **Predicts delivery time** based on:
  - Product category (Electronics, Clothing, Groceries, Furniture)
  - Shipping method (Standard, Express, Overnight)
  - Domestic/International destinations
  - Order quantity (bulk/non-bulk)
- **High Accuracy**: MAE of **1.6 hours** on test data.
- **Synthetic Data**: Realistic dataset with Indian logistics rules.
- **User-Friendly**: Simple inputs and clear outputs.

---

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/NawazBiyawarwala/Timelytics
   cd Timelytics
   ```

2. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

## Model Training

- Algorithm: Gradient Boosting Regressor
- **Key Hyperparameters:**:

  - n_estimators=300
  - learning_rate=0.05
  - max_depth=7

## Deployment

Deploy to Streamlit Community Cloud:

`Push code to a GitHub repository.`

`Sign up at Streamlit Cloud.`

`Connect your repo and deploy app.py`

## Usage

1. Generate Synthetic Data:

```
python data/generate_data.py
```

2. Train the Model:

```
python train_model.py
```

3. Run the Streamlit App:

```
streamlit run app.py
```
