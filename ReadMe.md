# CSE 482 – Sentiment Analysis of Customer Reviews

## Setup
1. Install packages:
   py -m pip install -r requirements.txt
2. In VS Code, select **Python 3.12.10** as the notebook kernel.

## Project layout
- `data/raw/` – original review datasets (don't edit)
- `data/processed/` – `train.csv` and `test.csv` (created by `00`)
- `00_data_prep.ipynb` – loads, labels, and splits the data (already run)
- `01_wordlist.ipynb` – word-list (lexicon) model
- `02_naive_bayes.ipynb` – Naive Bayes model
- `03_logreg_embeddings.ipynb` – Logistic Regression + word embeddings
- `04_compare.ipynb` – compares all results

## Rules
- Load data from `data/processed/train.csv` and `test.csv`. Don't modify them.
- Train only on `train.csv`, score only on `test.csv`.
- Report accuracy, precision, recall, F1, and a confusion matrix.
- Save your scores to `results_0X.csv` with columns: model, accuracy, precision, recall, f1.