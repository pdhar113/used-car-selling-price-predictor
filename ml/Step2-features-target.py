import pandas as pd

# Load training data
train = pd.read_csv("train.csv")

# Separate features and target
X = train[['car_name', 'vehicle_age', 'km_driven', 'seller_type', 'fuel_type', 'transmission_type']]
y = train['selling_price']

print("Features (X):")
print(X.head())

print("\nTarget (y):")
print(y.head())
