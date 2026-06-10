from fastapi import FastAPI
import joblib
app = FastAPI()
loaded = joblib.load("tree_model.pkl")
@app.get('/')
def home():
    return {"message": "Welcome to the Decision Tree Classifier API!"}
from fastapi import Form

@app.post("/predict")
def predict(features: str = Form(...)):
        features = [float(x) for x in features.split(",")]
        prediction = loaded.predict([features])
        return {"prediction": prediction.tolist()}
        