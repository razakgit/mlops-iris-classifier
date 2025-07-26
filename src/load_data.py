# src/load_data.py
from sklearn.datasets import load_iris
import pandas as pd

def save_iris():
    data = load_iris(as_frame=True)
    df = data.frame
    df['target'] = data.target
    df.to_csv("data/iris.csv", index=False)
    print("Iris dataset saved to data/iris.csv")

if __name__ == "__main__":
    save_iris()
