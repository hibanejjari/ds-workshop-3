from flask import Flask, request, jsonify
import pickle
import numpy as np
from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target
class_names = iris.target_names  # ['setosa', 'versicolor', 'virginica']

# Train an MLP Neural Network model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
mlp_model = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=500, random_state=42)
mlp_model.fit(X_train, y_train)

# Calculate model accuracy
mlp_accuracy = mlp_model.score(X_test, y_test)

# Save the model
with open('mlp_model.pkl', 'wb') as file:
    pickle.dump(mlp_model, file)

@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Extract parameters from URL query string
        sepal_length = float(request.args.get('sepal_length'))
        sepal_width = float(request.args.get('sepal_width'))
        petal_length = float(request.args.get('petal_length'))
        petal_width = float(request.args.get('petal_width'))

        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # Load the trained model
        with open('mlp_model.pkl', 'rb') as file:
            model = pickle.load(file)

        # Make prediction
        predicted_class = model.predict(features)[0]
        class_name = class_names[predicted_class]

        response = {
            "class_name": class_name,
            "predicted_class": int(predicted_class),
            "mlp_accuracy": round(mlp_accuracy, 2)  # Accuracy rounded to 2 decimal places
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/')
def home():
    return "MLP Neural Network Flask API is running!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
