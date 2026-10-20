from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "sample.csv"
MODEL_PATH = ROOT / "artifacts" / "model.joblib"


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    if not {"text", "label"}.issubset(data.columns):
        raise ValueError("The CSV must contain 'text' and 'label' columns.")
    if data[["text", "label"]].isna().any().any():
        raise ValueError("The dataset contains empty text or label values.")
    if data["label"].nunique() < 2 or data["label"].value_counts().min() < 2:
        raise ValueError("Each label needs at least two examples for a stratified split.")

    x_train, x_test, y_train, y_test = train_test_split(
        data["text"],
        data["label"],
        test_size=0.25,
        random_state=42,
        stratify=data["label"],
    )
    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True)),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    print(f"Holdout accuracy: {accuracy_score(y_test, predictions):.3f}")
    print(classification_report(y_test, predictions, zero_division=0))
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved demo model to {MODEL_PATH.relative_to(ROOT)}")
    print("Reminder: this tiny synthetic dataset is for demonstration only.")


if __name__ == "__main__":
    main()
