# train_model.py
import load_data
from load_data import iris_data
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def train_and_evaluate_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train.values, y_train.values)  # Use .values to convert to NumPy arrays

    y_pred = model.predict(X_test.values)
    accuracy = accuracy_score(y_test.values, y_pred)  # Use .values for y as well
    print("Model Accuracy:", accuracy)

    return model

X = iris_data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = iris_data['species']

# Example usage:
trained_model = train_and_evaluate_model(X, y)

