import argparse
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "model.joblib"


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify one text with the demo model.")
    parser.add_argument("text", help="Text to classify")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise SystemExit("Demo model not found. Run python src/train.py first.")

    model = joblib.load(MODEL_PATH)
    label = model.predict([args.text])[0]
    probabilities = model.predict_proba([args.text])[0]
    confidence = max(probabilities)
    print(f"Demo label: {label} (model score: {confidence:.2f})")
    print("This experimental score is not a measure of certainty or a moderation decision.")


if __name__ == "__main__":
    main()
