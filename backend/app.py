from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
from pathlib import Path

app = Flask(__name__)
CORS(app)  # Enable cross-origin requests for React frontend

# Load model and scaler using pathlib to avoid invalid escape sequences
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'model' / 'house_price_model.pkl'
SCALER_PATH = BASE_DIR / 'model' / 'scaler.pkl'

with MODEL_PATH.open('rb') as f:
    model = pickle.load(f)

with SCALER_PATH.open('rb') as f:
    scaler = pickle.load(f)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array([[data['bedrooms'], data['bathrooms'], data['livingArea'], 
                          data['condition'], data['schoolsNearby']]])
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)[0]
    return jsonify({'predicted_price': prediction*9})


if __name__ == '__main__':
    app.run(debug=True)
