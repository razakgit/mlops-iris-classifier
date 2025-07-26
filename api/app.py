from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import logging
from prometheus_fastapi_instrumentator import Instrumentator

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

@app.get("/")
def read_root():
    return {"message": "Iris Classifier API is up"}

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
