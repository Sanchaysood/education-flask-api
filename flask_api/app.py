from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load models from 'model/' directory relative to this file
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
kmeans = joblib.load(os.path.join(MODEL_DIR, "kmeans_model.pkl"))
gmm = joblib.load(os.path.join(MODEL_DIR, "gmm_model.pkl"))

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json["features"]
        scaled = scaler.transform([data])
        kmeans_result = int(kmeans.predict(scaled)[0])
        gmm_result = int(gmm.predict(scaled)[0])

        return jsonify({
            "kmeans_cluster": kmeans_result,
            "gmm_cluster": gmm_result
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# For Render deployment: run on 0.0.0.0 and use $PORT
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
