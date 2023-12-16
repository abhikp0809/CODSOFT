# predict_species.py
import load_data
import train_model
from load_data import iris_data
from train_model import trained_model
def predict_species(model):
    sepal_length = float(input("Enter sepal length (range 4:8): "))
    sepal_width = float(input("Enter sepal width(range 2:4): "))
    petal_length = float(input("Enter petal length (range 1:8): "))
    petal_width = float(input("Enter petal width (range 0:3): "))

    user_input = [[sepal_length, sepal_width, petal_length, petal_width]]
    predicted_species = model.predict(user_input)
    print("Predicted Species:", predicted_species[0])

predict_species(trained_model)
