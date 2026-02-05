import pandas as pd
from sklearn.linear_model import LinearRegression

# Load training data
train = pd.read_csv("train.csv")

# Separate features and target
X = train[['car_name', 'vehicle_age', 'km_driven', 'seller_type', 'fuel_type', 'transmission_type']]
y = train['selling_price']

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Display learned parameters
print("Intercept (Base Selling Price):", int(model.intercept_))
print("Coefficient for Car Name:", int(model.coef_[0]))
print("Coefficient for Vehicle Age:", int(model.coef_[1]))
print("Coefficient for Kilometer Driven:", int(model.coef_[2]))
print("Coefficient for Seller Type:", int(model.coef_[3]))
print("Coefficient for Fuel Type:", int(model.coef_[4]))
print("Coefficient for Transmission Type:", int(model.coef_[5]))
