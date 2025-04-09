from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Load the model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

# Define the input format
class HeartData(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

@app.get("/")
def home():
    return {"message": "Heart Disease Prediction API is live!"}

@app.post("/predict")
def predict(data: HeartData):
    input_data = np.array([[
        data.age, data.sex, data.cp, data.trestbps, data.chol,
        data.fbs, data.restecg, data.thalach, data.exang,
        data.oldpeak, data.slope, data.ca, data.thal
    ]])
    prediction = model.predict(input_data)
    result = "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease"
    return {"prediction": result}
