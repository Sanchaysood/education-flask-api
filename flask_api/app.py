from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

# Ensure working directory is correct when deployed
os.chdir(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)

# Load models from local 'model' folder
scaler = joblib.load("model/scaler.pkl")
kmeans = joblib.load("model/kmeans_model.pkl")
gmm = joblib.load("model/gmm_model.pkl")

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

# Required for Render deployment
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
