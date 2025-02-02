from fastapi import FastAPI
import pickle
import joblib
import pandas as pd
from data_model import Review

app = FastAPI(
    title="Sentiment Analysis",
    description="Sentiment prediction using Logistic Regression"
)

# Load the trained model
with open("logistic_regression_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the TF-IDF vectorizer
vectorizer = joblib.load("tfidf_vectorizer.pkl")

@app.get("/")
def index():
    return {"message": "Welcome to Sentiment Analysis API"}

@app.post("/predict")
def model_predict(review: Review):
    """Predict sentiment based on user input review."""
    
    # Convert input text into DataFrame
    sample_df = pd.DataFrame({"cleaned_review": [review.cleaned_review]})

    # Transform input using the loaded vectorizer
    sample_tfidf = vectorizer.transform(sample_df["cleaned_review"])

    # Predict sentiment
    predicted_label = model.predict(sample_tfidf)[0]  # Extract the single value
    
    return {"predicted_sentiment": predicted_label}


