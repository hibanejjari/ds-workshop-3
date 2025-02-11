from flask import Flask, request, jsonify
import requests
import numpy as np


app = Flask(__name__)
SVM_API_URL = "https://9d0d-2001-861-5865-6b10-b8b0-7cd0-d0e0-ed26.ngrok-free.app/predict"
RF_API_URL = "https://your_random_forest_ngrok_url/predict"  # Replace with RF URL


@app.route('/predict', methods=['GET'])
def aggregate_prediction():
    try:
        sepal_length = float(request.args.get('sepal_length'))
        sepal_width = float(request.args.get('sepal_width'))
        petal_length = float(request.args.get('petal_length'))
        petal_width = float(request.args.get('petal_width'))

        params = {
            "sepal_length": sepal_length,
            "sepal_width": sepal_width,
            "petal_length": petal_length,
            "petal_width": petal_width
        }

        # Call SVM API
        svm_response = requests.get(SVM_API_URL, params=params)
        if svm_response.status_code != 200:
            return jsonify({"error": "SVM API failed"}), 500
        svm_data = svm_response.json()

        # Call Random Forest API
        rf_response = requests.get(RF_API_URL, params=params)
        if rf_response.status_code != 200:
            return jsonify({"error": "Random Forest API failed"}), 500
        rf_data = rf_response.json()

        # Extract predictions
        svm_pred = svm_data.get("predicted_class")
        rf_pred = rf_data.get("Random Forest Prediction", {}).get("rf_prediction")

        if svm_pred is None or rf_pred is None:
            return jsonify({"error": "Missing predictions from models"}), 500

        # Compute consensus
        predictions = [svm_pred, rf_pred]
        consensus_prediction = int(np.round(np.mean(predictions)))
        consensus_class = svm_data["class_name"] if consensus_prediction == svm_pred else rf_data["Random Forest Prediction"]["rf_species"]

        return jsonify({
            "consensus_prediction": consensus_prediction,
            "consensus_class": consensus_class,
            "individual_predictions": {
                "svm": svm_data,
                "random_forest": rf_data
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=6000, debug=True)
