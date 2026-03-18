from flask import Flask, render_template, jsonify, request, send_file
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import os
from datetime import datetime, timedelta

app = Flask(__name__)

# --- BULLETPROOF PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "sensex_data_perfect.csv")
MODEL_FILE = os.path.join(BASE_DIR, "model_data", "sensex_model.keras") 
SCALER_FILE = os.path.join(BASE_DIR, "model_data", "price_scaler.pkl")
WINDOW_SIZE = 60

# Load AI safely
try:
    model = load_model(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    print("[INFO] AI Brain Loaded Successfully.")
except Exception as e:
    print(f"[WARNING] AI Model not loaded. Error: {e}")
    model, scaler = None, None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/data', methods=['GET'])
def get_data():
    days = int(request.args.get('days', 100))
    try:
        df = pd.read_csv(DATA_FILE)
        chart_data = df.tail(days).replace({np.nan: None})
        return jsonify(chart_data.to_dict(orient='records'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict', methods=['GET'])
def predict():
    if model is None or scaler is None:
        return jsonify({"error": "AI Model is down."}), 500

    try:
        df = pd.read_csv(DATA_FILE)
        last_60_days = df['Close'].tail(WINDOW_SIZE).values
        last_price = float(last_60_days[-1])
        last_date = str(df['Date'].iloc[-1]).split(' ')[0]
        
        last_60_days_scaled = scaler.transform(last_60_days.reshape(-1, 1))
        X_input = np.array([last_60_days_scaled]).reshape(1, WINDOW_SIZE, 1)
        
        predicted_decimal = model.predict(X_input, verbose=0)
        predicted_price = float(scaler.inverse_transform(predicted_decimal)[0][0])
        
        return jsonify({
            "target_date": "Tomorrow",
            "last_price": round(last_price, 2),
            "predicted_price": round(predicted_price, 2),
            "difference": round(predicted_price - last_price, 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict_custom', methods=['POST'])
def predict_custom():
    """Advanced Scenario Testing using all Trainer parameters"""
    if model is None or scaler is None:
        return jsonify({"error": "AI Model is down."}), 500

    try:
        data = request.get_json()
        
        # 1. Extract the Trainer's hypothetical inputs
        custom_date_str = data.get('date', '')
        custom_close = float(data.get('close', 0))
        
        # Calculate the "Next Day" date
        try:
            date_obj = datetime.strptime(custom_date_str, "%Y-%m-%d")
            next_day_obj = date_obj + timedelta(days=1)
            target_date = next_day_obj.strftime("%Y-%m-%d")
        except:
            target_date = "Next Trading Day"

        # 2. Extract Univariate sequence (Close Price)
        df = pd.read_csv(DATA_FILE)
        last_59_days = df['Close'].tail(WINDOW_SIZE - 1).values
        
        # 3. Append the Trainer's custom Close to complete the 60-day window
        custom_60_days = np.append(last_59_days, custom_close)
        
        # 4. Predict
        custom_60_scaled = scaler.transform(custom_60_days.reshape(-1, 1))
        X_input = np.array([custom_60_scaled]).reshape(1, WINDOW_SIZE, 1)
        
        predicted_decimal = model.predict(X_input, verbose=0)
        predicted_price = float(scaler.inverse_transform(predicted_decimal)[0][0])
        
        return jsonify({
            "target_date": target_date,
            "last_price": round(custom_close, 2),
            "predicted_price": round(predicted_price, 2),
            "difference": round(predicted_price - custom_close, 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/download')
def download_csv():
    return send_file(DATA_FILE, as_attachment=True, download_name="sensex_data_perfect.csv")

if __name__ == '__main__':
    app.run(debug=True, port=5000)