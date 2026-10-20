# Cyberbullying Text Classification Demo

> A new portfolio demo inspired by the project description in my resume. This is a small, independent implementation; it is not the original academic project or its trained BERT model.

This command-line demo trains a text classifier on a tiny synthetic dataset, reports a held-out evaluation, and predicts whether new text resembles the demo's bullying or non_bullying examples.

## Requirements

- Python 3.10 or newer

## Run

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/train.py
python src/predict.py "You do not belong here"
```

Training creates `artifacts/model.joblib`. The script prints accuracy, precision, recall, and F1 for a stratified holdout split. These numbers are only a software demonstration: the included examples are synthetic and far too small to support real-world conclusions.

## Dataset format

`data/sample.csv` shows the required columns: `text` and `label`. To explore other data, use a properly licensed, representative dataset and keep private or identifying information out of the repository. Review and document the dataset's source and permitted use before adding it.

## Limitations and responsible use

- This is a simple TF-IDF and logistic regression baseline, not a fine-tuned BERT model.
- Synthetic examples do not represent the range of language, context, dialects, or reclaimed language found online. The classifier can be wrong and may reflect bias in its examples.
- Do not use this demo to moderate real people, make disciplinary decisions, or replace human review.
- It does not provide legal advice or crisis support. Any support links for a real deployment must be verified for the intended region and kept current.

## Project structure

```text
data/sample.csv       Tiny synthetic demonstration data
src/train.py          Train and evaluate the baseline
src/predict.py        Predict a label for one text input
requirements.txt      Python dependencies
```
