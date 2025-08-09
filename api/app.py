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
import datetime

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
    # Load original data and feedback data
    orig_df = pd.read_csv("data/iris.csv")
    if os.path.exists("feedback.csv"):
        feedback_df = pd.read_csv("feedback.csv")
        # Find feedback rows not present in original data
        new_feedback = pd.concat([feedback_df, orig_df]).drop_duplicates(keep=False)
        print("Additional feedback received (not in original iris.csv):")
        print(new_feedback)
    else:
        print("No feedback.csv found.")
    return {"message": "Feedback received"}

@app.get("/")
def read_root():
    current_time = datetime.datetime.now().isoformat()
    return {"message": "Iris Classifier API is up1", "time": current_time}

@app.get("/metrics")
def metrics():
    Instrumentator().instrument(app).expose(app)
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

@app.post("/predict")
def predict(input: IrisInput):
    data = pd.DataFrame([[
        input.sepal_length,
        input.sepal_width,
        input.petal_length,
        input.petal_width
    ]], columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])
    # Reload model on each prediction request
    model_reloaded = False
    if os.path.exists("models/best_model.pkl"):
        model = joblib.load("models/best_model.pkl")
        model_reloaded = True
    else:
        model = None

    if model is not None:
        print(f"Model is loaded. Reloaded: {model_reloaded}")
        prediction = model.predict(data)[0]
    else:
        print("Model is NOT loaded.")
        prediction = None
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

    # Print additional feedback data not in original
    new_feedback = pd.concat([feedback_df, orig_df]).drop_duplicates(keep=False)
    print("Additional feedback received (not in original iris.csv):")
    print(new_feedback)

    X = combined_df.drop("target", axis=1)
    y = combined_df["target"]

    model = RandomForestClassifier()
    model.fit(X, y)

    dump(model, "models/best_model.pkl")
    model = joblib.load("models/best_model.pkl")

    #dump(model, "model.joblib")  # overwrite existing model
    #model = load("model.joblib")  # reload into memory

    return {"message": "Model retrained with feedback data."}
