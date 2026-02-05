from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

MODEL_PATH = "used_car_selling_price_model.pkl"
TEST_PATH = "test.csv"
ORIG_DATASET_PATH = "orig-dataset.csv"
ENCODED_DATASET_PATH = "encoded-dataset.csv"

FEATURE_COLUMNS = [
    "car_name",
    "vehicle_age",
    "km_driven",
    "seller_type",
    "fuel_type",
    "transmission_type",
]

MAX_VEHICLE_AGE = 29
MAX_KM_DRIVEN = 4000000


def load_car_options():
    try:
        orig = pd.read_csv(ORIG_DATASET_PATH, usecols=["car_name"])
        enc = pd.read_csv(ENCODED_DATASET_PATH, usecols=["car_name"])
        if len(orig) != len(enc):
            return []

        df = pd.DataFrame({"car_name": orig["car_name"], "car_code": enc["car_name"]})
        df = df.drop_duplicates(subset=["car_name"]).copy()
        df["car_code"] = df["car_code"].astype(int)
        df = df.sort_values("car_name")
        return df.to_dict("records")
    except Exception:
        return []


# -----------------------------
# Load trained model ONCE
# -----------------------------
model = joblib.load(MODEL_PATH)

# -----------------------------
# Load test data for confidence
# -----------------------------
model_confidence = None
try:
    test = pd.read_csv(TEST_PATH)
    X_test = test[FEATURE_COLUMNS]
    y_test = test["selling_price"]
    model_confidence = round(model.score(X_test, y_test) * 100, 2)
except Exception:
    model_confidence = None

CAR_OPTIONS = load_car_options()
CAR_CODE_SET = {option["car_code"] for option in CAR_OPTIONS}


# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html", car_options=CAR_OPTIONS, error=None)


@app.route("/predict", methods=["POST"])
def predict():
    error_message = None

    try:
        car_name_val = int(request.form.get("car_name", ""))
        vehicle_age_val = int(request.form.get("vehicle_age", ""))
        km_driven_val = int(request.form.get("km_driven", ""))
        seller_type_val = int(request.form.get("seller_type", ""))
        fuel_type_val = int(request.form.get("fuel_type", ""))
        transmission_type_val = int(request.form.get("transmission_type", ""))

        if CAR_CODE_SET and car_name_val not in CAR_CODE_SET:
            raise ValueError("Please select a valid car model.")
        if vehicle_age_val < 0 or vehicle_age_val > MAX_VEHICLE_AGE:
            raise ValueError("Vehicle age must be between 0 and 29 years.")
        if km_driven_val < 0 or km_driven_val > MAX_KM_DRIVEN:
            raise ValueError("Kilometers driven must be between 0 and 4,000,000.")

        payload = [[
            car_name_val,
            vehicle_age_val,
            km_driven_val,
            seller_type_val,
            fuel_type_val,
            transmission_type_val,
        ]]

        predicted_price = model.predict(payload)[0]

        return render_template(
            "result.html",
            price=int(predicted_price),
            confidence=model_confidence,
        )
    except ValueError as exc:
        error_message = str(exc)
    except Exception:
        error_message = "Something went wrong. Please double-check your inputs and try again."

    return render_template("index.html", car_options=CAR_OPTIONS, error=error_message)


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")
