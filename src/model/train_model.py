import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import mlflow
import mlflow.sklearn
import joblib
import pickle


def train_logistic_regression(X_train, y_train, output_dir):
    """Train Logistic Regression for sentiment analysis and save the model."""
    print("Training Logistic Regression model...")

    # Convert text to TF-IDF vectors
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train["cleaned_review"])
    
    # Define and train Logistic Regression model
    model = LogisticRegression()

    
    model.fit(X_train_tfidf, y_train)
    print(f"Loaded model type: {type(model)}")
    
    print(f"Model trained successfully!")

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the trained model and vectorizer
    with open("logistic_regression_model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    joblib.dump(vectorizer, output_dir / "tfidf_vectorizer.pkl")

    return model, vectorizer

if __name__ == "__main__":
    # Define paths
    data_dir = Path("data/processed")
    output_dir = Path("./")

    # Load processed data
    X_train = pd.read_csv(data_dir / "X_train.csv")
    y_train = pd.read_csv(data_dir / "y_train.csv").values.ravel()  # Convert to 1D array

    # Train and save the model
    train_logistic_regression(X_train, y_train, output_dir)
