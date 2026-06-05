from flask import Flask, request, jsonify
import pandas as pd
import pickle
import time
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

app = Flask(__name__)

# Metrik Prometheus
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('http_request_latency_seconds', 'Request latency', ['method', 'endpoint'])
PREDICTION_COUNT = Counter('prediction_requests_total', 'Total prediction requests')
PREDICTION_ERROR = Counter('prediction_errors_total', 'Total prediction errors')
MODEL_LOADED = Gauge('model_loaded', 'Whether model is loaded (1 for yes, 0 for no)')

# Load model dari file pickle
try:
    with open('random_forest_model.pkl', 'rb') as f:
        model = pickle.load(f)
    MODEL_LOADED.set(1)
    print("Model loaded from random_forest_model.pkl")
except Exception as e:
    MODEL_LOADED.set(0)
    print(f"Failed to load model: {e}")
    model = None

@app.route('/health', methods=['GET'])
def health():
    REQUEST_COUNT.labels(method='GET', endpoint='/health', status='200').inc()
    return jsonify({"status": "ok"}), 200

@app.route('/metrics', methods=['GET'])
def metrics():
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    REQUEST_COUNT.labels(method='POST', endpoint='/predict', status='200').inc()
    PREDICTION_COUNT.inc()
    if model is None:
        PREDICTION_ERROR.inc()
        return jsonify({"error": "Model not loaded"}), 500
    try:
        data = request.get_json(force=True)
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = pd.DataFrame(data)
        prediction = model.predict(df)[0]
        latency = time.time() - start_time
        REQUEST_LATENCY.labels(method='POST', endpoint='/predict').observe(latency)
        return jsonify({"prediction": int(prediction)})
    except Exception as e:
        PREDICTION_ERROR.inc()
        REQUEST_COUNT.labels(method='POST', endpoint='/predict', status='500').inc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)