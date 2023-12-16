# load_data.py
import pandas as pd

def load_dataset(file_path):
    iris_data = pd.read_csv(file_path)
    return iris_data

def display_initial_data(dataset):
    print(dataset.head())

# Example usage:
iris_data = load_dataset('IRIS.csv')
display_initial_data(iris_data)
