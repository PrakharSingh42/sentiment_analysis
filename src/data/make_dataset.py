from pathlib import Path
import nltk
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


def load_data(file_path: Path) -> pd.DataFrame:
    file_extension = file_path.suffix  # Get file extension

    try:
        if file_extension == ".csv":
            return pd.read_csv(file_path, encoding="utf-8")
        elif file_extension == ".txt":
            return pd.read_table(file_path)
        elif file_extension == ".xlsx":
            return pd.read_excel(file_path)
        elif file_extension == ".json":
            return pd.read_json(file_path)
        else:
            raise ValueError("Unsupported file format")
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return pd.DataFrame()


def validate_data(data: pd.DataFrame) -> pd.DataFrame:
    data.dropna(inplace=True)  # Remove rows with missing values
    data.drop_duplicates(inplace=True)  # Remove duplicate rows
    return data


def clean_text(text: str) -> str:
    if not isinstance(text, str):  # Ensure input is a string
        return ""

    text = re.sub(r"<.*?>", " ", text)  # Remove HTML tags
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = text.lower()

    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return " ".join(tokens)


def preprocess_text(data: pd.DataFrame) -> pd.DataFrame:
    data["cleaned_review"] = data["review"].apply(lambda x: clean_text(x))
    return data


def split_data(data: pd.DataFrame, output_dir: Path) -> None:
    X = data["cleaned_review"]
    y = data["sentiment"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    # Create directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save splits
    X_train.to_csv(output_dir / "X_train.csv", index=False)
    X_val.to_csv(output_dir / "X_val.csv", index=False)
    X_test.to_csv(output_dir / "X_test.csv", index=False)
    y_train.to_csv(output_dir / "y_train.csv", index=False)
    y_val.to_csv(output_dir / "y_val.csv", index=False)
    y_test.to_csv(output_dir / "y_test.csv", index=False)

    print(f"Data splits saved in {output_dir}")


def main():
    """
    Main function to execute the data pipeline.
    """
    input_file = Path("data/raw/IMDB_Dataset.csv")  # Use Path object
    output_dir = Path("data/processed")  # Use Path object

    # Step-by-step pipeline execution
    data = load_data(input_file)
    
    if data.empty:
        print("Error: No data loaded. Exiting pipeline.")
        return

    data = validate_data(data)
    data = preprocess_text(data)
    split_data(data, output_dir)

    print(data.info())
    print(data.head())
    print("Data pipeline executed successfully!")


if __name__ == "__main__":
    main()
