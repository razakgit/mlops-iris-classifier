# src/load_data.py
from sklearn.datasets import load_iris
import pandas as pd

def save_iris():
    data = load_iris(as_frame=True)
    df = data.frame
    # Rename columns to match the expected format
    df = df.rename(columns={
        "sepal length (cm)": "sepal_length",
        "sepal width (cm)": "sepal_width",
        "petal length (cm)": "petal_length",
        "petal width (cm)": "petal_width"
    })
    df['target'] = data.target
    df.to_csv("data/iris.csv", index=False)
    print("Iris dataset saved to data/iris.csv")

if __name__ == "__main__":
    save_iris()
