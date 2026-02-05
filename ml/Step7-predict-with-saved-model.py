import joblib

# Load trained model
model = joblib.load("used_car_selling_price_model.pkl")

# New input (simulating user input)
carName = 50       
vAge = 8 
kmDriven = 20000
sellerType = 0
fuelType = 0
transmissionType = 1

prediction = model.predict([[carName, vAge, kmDriven, sellerType, fuelType, transmissionType]])

print("Input:")
print("Car Name:", carName)
print("Vehicle Age:", vAge, " Years")
print("Kilometer Driven:", kmDriven, "  Kilometers")
print("Seller Type:", sellerType)
print("Fuel Type:", fuelType)
print("Transmission Type:", transmissionType)


print("Predicted Selling Price (INR):", int(prediction[0]))
