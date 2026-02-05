from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# -----------------------------
# Load trained model ONCE
# -----------------------------
model = joblib.load("used_car_selling_price_model.pkl")

# -----------------------------
# Load test data for confidence
# -----------------------------
test = pd.read_csv("test.csv")
X_test = test[['car_name', 'vehicle_age', 'km_driven', 'seller_type', 'fuel_type', 'transmission_type']]
y_test = test['selling_price']

model_confidence = round(model.score(X_test, y_test) * 100, 2)

# -----------------------------
# Routes
# -----------------------------
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    carNameVal = int(request.form['car_name'])
    vehicleAgeVal = int(request.form['vehicle_age'])
    kmDrivenVal = int(request.form['km_driven'])
    sellerTypeVal = int(request.form['seller_type'])
    fuelTypeVal = int(request.form['fuel_type'])
    transmissionTypeVal = int(request.form['transmission_type'])


    predicted_price = model.predict([[carNameVal, vehicleAgeVal, kmDrivenVal, sellerTypeVal, fuelTypeVal, transmissionTypeVal]])[0]

    return render_template(
        "result.html",
        price=int(predicted_price),
        confidence=model_confidence
    )

if __name__ == "__main__":
    app.run(debug=True)