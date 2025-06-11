from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re

@app.get("/")
def root():
    return {"message": "Expense Categorization API is working. Use /predict."}


# 1. Start the app
app = FastAPI()

# 2. Load the model and vectorizer
model = joblib.load("sms_categorization_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# 3. Define the structure of input data
class SMSInput(BaseModel):
    description: str  # this is the text input

# 4. Text preprocessing (same as training)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = ' '.join(text.split())
    return text

# 5. API endpoint: when POST request is made to /predict
@app.post("/predict")
def predict_category(input_data: SMSInput):
    # Preprocess
    processed = preprocess_text(input_data.description)
    # Transform with vectorizer
    vector = vectorizer.transform([processed])
    # Predict
    prediction = model.predict(vector)[0]
    confidence = max(model.predict_proba(vector)[0])
    # Return as JSON
    return {
        "category": prediction,
        "confidence": round(confidence, 2)
    }
