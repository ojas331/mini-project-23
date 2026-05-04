from flask import Flask, request, jsonify
from flask_cors import CORS
import math
import random
import os
import pickle
import numpy as np
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Attempt to load model and scaler globally
MODEL_PATH = 'models/nimbus_lstm_v1.keras'
SCALER_PATH = 'models/nimbus_scaler_v1.pkl'

try:
    from tensorflow.keras.models import load_model
    model = load_model(MODEL_PATH)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    print("[API] Successfully loaded Keras model and Scaler.")
except Exception as e:
    print(f"[API] Warning: Could not load model/scaler: {e}")
    model = None
    scaler = None

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
            
        base_temp = float(data.get('temp', 31.0))
        hum = float(data.get('hum', 64.0))
        pres = float(data.get('pres', 1009.0))
        aqi = float(data.get('aqi', 520.0))
        hour = float(data.get('hour', 14.0))

        forecast = []

        if model is not None and scaler is not None:
            try:
                # Prepare a sequence of 20 steps (required by LSTM)
                seq = []
                for i in range(20):
                    h = (hour - 10 + i * 0.5) % 24
                    d_sin = math.sin(2 * math.pi * 120 / 365)
                    d_cos = math.cos(2 * math.pi * 120 / 365)
                    seq.append([
                        base_temp, hum, pres, aqi,
                        math.sin(2 * math.pi * h / 24), math.cos(2 * math.pi * h / 24),
                        d_sin, d_cos,
                        0.0, 0.0  # temp_diff, humid_diff
                    ])
                
                seq_scaled = scaler.transform(seq)
                X = seq_scaled.reshape(1, 20, 10)
                
                # Predict 48 steps (30 min intervals)
                preds = model.predict(X, verbose=0)[0]
                
                # Aggregate 48 steps into 5 days
                for day in range(1, 6):
                    # Each day is approx 9-10 steps (48 / 5)
                    day_preds = preds[(day-1)*9 : day*9]
                    if len(day_preds) == 0:
                        day_preds = [base_temp]
                        
                    day_min = float(np.min(day_preds))
                    day_max = float(np.max(day_preds))
                    day_avg = float(np.mean(day_preds))
                    
                    date = datetime.now() + timedelta(days=day)
                    forecast.append({
                        'day': date.strftime('%a, %d %b'),
                        'min': round(day_min, 1),
                        'max': round(day_max, 1),
                        'avg': round(day_avg, 1)
                    })
                
                return jsonify({
                    'source': 'trained_model',
                    'forecast': forecast
                })
            except Exception as e:
                print(f"[API] Error predicting with model: {e}")
                # Fallback to simulation below

        # Fallback simulation
        current_temp = base_temp
        for day in range(1, 6):
            daily_variation = math.sin(2 * math.pi * (hour + day * 24) / 24) * 3
            noise = random.uniform(-1, 1)
            hum_effect = (hum - 50) * -0.05
            pres_effect = (1013 - pres) * 0.02
            
            day_avg = current_temp + daily_variation + hum_effect + pres_effect + noise
            day_max = day_avg + random.uniform(2, 4)
            day_min = day_avg - random.uniform(2, 4)
            
            date = datetime.now() + timedelta(days=day)
            
            forecast.append({
                'day': date.strftime('%a, %d %b'),
                'min': round(day_min, 1),
                'max': round(day_max, 1),
                'avg': round(day_avg, 1)
            })
            current_temp = current_temp * 0.9 + 29.0 * 0.1

        return jsonify({
            'source': 'simulation',
            'forecast': forecast
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("[API] Starting LSTM Model API on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
