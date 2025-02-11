from flask import Flask, request, jsonify
import numpy as np
import pickle
from sklearn import datasets

app = Flask(__name__)

# Load Iris dataset to retrieve class names
iris = datasets.load_iris()

# Load the pre-trained SVM model
with open("svm_iris_model.pkl", "rb") as f:
    data = pickle.load(f)
    model = data["model"]
    accuracy = data["accuracy"]  # Stored accuracy

@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Get input parameters from URL
        sepal_length = float(request.args.get('sepal_length', 5.1))
        sepal_width = float(request.args.get('sepal_width', 3.5))
        petal_length = float(request.args.get('petal_length', 1.4))
        petal_width = float(request.args.get('petal_width', 0.2))

        # Prepare input for prediction
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # Make prediction
        prediction = model.predict(input_data)[0]  
        class_name = iris.target_names[prediction]  

        response = {
            "predicted_class": int(prediction),
            "class_name": class_name,
            "svm_accuracy": round(accuracy, 2)
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 400
@app.route('/')
def home():
    return "SVM API is running!"


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5002, debug=True)
