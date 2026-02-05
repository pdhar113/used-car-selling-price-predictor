import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# -----------------------------
# Load training data
# -----------------------------
train = pd.read_csv("train.csv")
X_train = train[['car_name', 'vehicle_age', 'km_driven', 'seller_type', 'fuel_type', 'transmission_type']]
y_train = train['selling_price']

# -----------------------------
# Train model
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# Save trained model
# -----------------------------
joblib.dump(model, "used_car_selling_price_model.pkl")

print("Model saved as used_car_selling_price_model.pkl")
