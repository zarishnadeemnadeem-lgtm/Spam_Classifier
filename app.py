from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

# Updated file names as requested
vectorizer = joblib.load('tfidf_vectorizer.pkl')
model = joblib.load('spam_model .pkl')

class TextPayload(BaseModel):
    text: str

@app.post("/predict")
def predict(data: TextPayload):
    # 1. Text ko vectorizer se transform karein
    X = vectorizer.transform([data.text])
    
    # 2. Prediction praapt karein (0 = ham, 1 = spam)
    prediction = int(model.predict(X)[0])
    probabilities = model.predict_proba(X)[0].tolist()
    
    label = "spam" if prediction == 1 else "ham"
    
    return {
        "text": data.text,
        "prediction": prediction,
        "label": label,
        "probability_ham": probabilities[0],
        "probability_spam": probabilities[1]
    }