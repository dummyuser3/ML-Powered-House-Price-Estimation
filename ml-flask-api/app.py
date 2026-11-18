"""Flask REST API for house price predictions."""

from pathlib import Path
from math import isfinite

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from werkzeug.exceptions import BadRequest, HTTPException, UnsupportedMediaType

FEATURES = ["area", "bedrooms", "age"]
MODEL_PATH = Path(__file__).resolve().with_name("house_price_model.pkl")

app = Flask(__name__)

try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    model = None


@app.get("/")
def home():
    """Return a short message confirming that the API is available."""
    return jsonify({"message": "Welcome to House Price Prediction API"})


@app.post("/predict")
def predict():
    """Validate a JSON request and return the model's predicted price."""
    try:
        payload = request.get_json()
    except (BadRequest, UnsupportedMediaType) as error:
        return jsonify({"error": error.description}), error.code

    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    missing_features = [name for name in FEATURES if name not in payload]
    if missing_features:
        return jsonify({
            "error": "Missing required features: " + ", ".join(missing_features)
        }), 400

    values = {}
    for name in FEATURES:
        value = payload[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return jsonify({"error": f"'{name}' must be a numeric value."}), 400
        try:
            numeric_value = float(value)
        except (TypeError, ValueError, OverflowError):
            return jsonify({"error": f"'{name}' must be a numeric value."}), 400
        if not isfinite(numeric_value):
            return jsonify({"error": f"'{name}' must be a finite number."}), 400
        values[name] = numeric_value

    if values["area"] <= 0:
        return jsonify({"error": "'area' must be greater than zero."}), 400
    if values["bedrooms"] <= 0 or not values["bedrooms"].is_integer():
        return jsonify({"error": "'bedrooms' must be a positive whole number."}), 400
    if values["age"] < 0:
        return jsonify({"error": "'age' cannot be negative."}), 400

    if model is None:
        return jsonify({
            "error": "The trained model is missing. Run 'python train_model.py' first."
        }), 503

    features = pd.DataFrame([values], columns=FEATURES)
    prediction = float(model.predict(features)[0])
    return jsonify({"prediction": round(prediction, 2)})


@app.errorhandler(HTTPException)
def handle_http_error(error):
    """Return Flask/Werkzeug HTTP errors as JSON instead of HTML."""
    return jsonify({"error": error.description}), error.code


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
