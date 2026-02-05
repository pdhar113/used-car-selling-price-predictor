import pandas as pd
from sklearn.linear_model import LinearRegression

# -----------------------------
# Load training data
# -----------------------------
train = pd.read_csv("train.csv")
X_train = train[['car_name', 'vehicle_age', 'km_driven', 'seller_type', 'fuel_type', 'transmission_type']]
y_train = train['selling_price']

# -----------------------------
# Train the model
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# Load test data
# -----------------------------
test = pd.read_csv("test.csv")
X_test = test[['car_name', 'vehicle_age', 'km_driven', 'seller_type', 'fuel_type', 'transmission_type']]
y_test = test['selling_price']

# -----------------------------
# Evaluate confidence on test data
# -----------------------------
confidence = model.score(X_test, y_test)
confidence_percent = round(confidence * 100, 2)

print("Model Confidence on UNSEEN Test Data:", confidence_percent, "%")
