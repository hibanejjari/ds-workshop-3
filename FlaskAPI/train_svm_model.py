import pickle
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Load Iris dataset
iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Train an SVM model
svm_model = SVC(probability=True)
svm_model.fit(X_train, y_train)

# Calculate accuracy
accuracy = svm_model.score(X_test, y_test)

# Save the model and accuracy
with open("svm_iris_model.pkl", "wb") as file:
    pickle.dump({"model": svm_model, "accuracy": accuracy}, file)

print("✅ SVM model trained and saved as 'svm_iris_model.pkl'")
