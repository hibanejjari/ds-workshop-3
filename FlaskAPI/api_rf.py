from flask import Flask, request, jsonify
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

@app.route('/')
def home():
    return "Hello, World!"

@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Get input parameters from URL
        sepal_length = float(request.args.get('sepal_length'))
        sepal_width = float(request.args.get('sepal_width'))
        petal_length = float(request.args.get('petal_length'))
        petal_width = float(request.args.get('petal_width'))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid input. Please provide all four features as valid numbers."}), 400

    # Make prediction
    rf_prediction = rf_model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    rf_species = iris.target_names[rf_prediction[0]]
    rf_accuracy = rf_model.score(X_test, y_test)

    return jsonify({
        "Random Forest Prediction": {
            "rf_accuracy": round(rf_accuracy, 2),
            "rf_prediction": int(rf_prediction[0]),
            "rf_species": rf_species
        }
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001, debug=True)

