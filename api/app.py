from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import logging
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import start_http_server, Summary, Counter
import pandas as pd
import os

from joblib import load, dump
from sklearn.ensemble import RandomForestClassifier

REQUEST_COUNT = Counter('request_count', 'Total number of requests')

app = FastAPI()
Instrumentator().instrument(app).expose(app)

# Load model
model = joblib.load("models/best_model.pkl")

# Logging
logging.basicConfig(filename="logs/prediction.log", level=logging.INFO)

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class FeedbackData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    target: int

@app.post("/feedback")
def collect_feedback(data: FeedbackData):
    df = pd.DataFrame([data.dict()])
    if not os.path.exists("feedback.csv"):
        df.to_csv("feedback.csv", index=False)
    else:
        df.to_csv("feedback.csv", mode='a', header=False, index=False)
    return {"message": "Feedback received"}

@app.get("/")
def read_root():
    return {"message": "Iris Classifier API is up"}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

@app.post("/predict")
def predict(input: IrisInput):
    data = [[
        input.sepal_length,
        input.sepal_width,
        input.petal_length,
        input.petal_width
    ]]
    prediction = model.predict(data)[0]
    logging.info(f"Input: {input.dict()}, Prediction: {prediction}")
    return {"prediction": int(prediction)}

@app.post("/retrain")
def retrain_model():
    # Load original training data

    orig_df = pd.read_csv("data/iris.csv")
    
    # Append feedback data
    if os.path.exists("feedback.csv"):
        feedback_df = pd.read_csv("feedback.csv")
        combined_df = pd.concat([orig_df, feedback_df], ignore_index=True)
    else:
        return {"message": "No feedback data to retrain on."}

 # Drop rows where label is NaN
    combined_df = combined_df.dropna(subset=["target"])

    X = combined_df.drop("target", axis=1)
    y = combined_df["target"]

    model = RandomForestClassifier()
    model.fit(X, y)

    dump(model, "models/best_model.pkl")
    model = joblib.load("models/best_model.pkl")

    #dump(model, "model.joblib")  # overwrite existing model
    #model = load("model.joblib")  # reload into memory

    return {"message": "Model retrained with feedback data."}
