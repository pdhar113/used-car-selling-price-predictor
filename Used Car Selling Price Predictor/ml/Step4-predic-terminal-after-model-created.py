import pandas as pd
from sklearn.linear_model import LinearRegression

# Load training data
train = pd.read_csv("train.csv")

# Separate features and target
X = train[['car_name', 'vehicle_age', 'km_driven', 'seller_type', 'fuel_type', 'transmission_type']]
y = train['selling_price']

# Train the model
model = LinearRegression()
model.fit(X, y)

# ---- Input for Prediction ----
carName = 86       
vAge = 8 
kmDriven = 20000
sellerType = 0
fuelType = 0
transmissionType = 1


prediction = model.predict([[carName, vAge, kmDriven, sellerType, fuelType, transmissionType]])

# -----------------------------
# Confidence (R² score)
# -----------------------------
confidence = model.score(X, y)   # on training data
confidence_percent = round(confidence * 100, 2)

print("Input:")
print("Car Name:", carName)
print("Vehicle Age:", vAge, " Years")
print("Kilometer Driven:", kmDriven, "  Kilometers")
print("Seller Type:", sellerType)
print("Fuel Type:", fuelType)
print("Transmission Type:", transmissionType)

print("\nPredicted Selling Price (INR):", int(prediction[0]))
print("Model Confidence Level:", confidence_percent, "%")
