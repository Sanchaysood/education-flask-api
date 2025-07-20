from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

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

if __name__ == "__main__":
    app.run(debug=True)
