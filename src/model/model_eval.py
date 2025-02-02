import pandas as pd
import json
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report
import joblib

def evaluate_model(X_test, y_test, model_path, vectorizer_path, metrics_path):
    """Load model, evaluate it on test data, and save metrics as JSON."""
    print("Loading model and vectorizer...")
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    # Convert text to TF-IDF vectors using the saved vectorizer
    X_test_tfidf = vectorizer.transform(X_test["cleaned_review"])

    # Predict and evaluate
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{report}")

    # Save metrics as a JSON file
    metrics = {
        "accuracy": accuracy,
        "classification_report": report
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Metrics saved in {metrics_path}")

    return accuracy, report

if __name__ == "__main__":
    # Define paths
    data_dir = Path("data/processed")
    model_dir = Path("./")
    metrics_path = Path("metrics.json")
    
    # Load processed test data
    X_test = pd.read_csv(data_dir / "X_test.csv")
    y_test = pd.read_csv(data_dir / "y_test.csv").values.ravel()  # Convert to 1D array

    # Load and evaluate the model
    evaluate_model(
        X_test, y_test,
        model_dir / "logistic_regression_model.pkl",
        model_dir / "tfidf_vectorizer.pkl",
        metrics_path
    )
