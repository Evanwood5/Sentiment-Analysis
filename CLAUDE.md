# CLAUDE.md: CSE 482 Sentiment Analysis

## Project
Team project for CSE 482 by Evan (GitHub: Evanwood5, repo owner), Adiva (adiva-jain) and Tanmay (TanmayDabhade).
Goal: classify customer-review sentences as **positive or negative**. Compare a word-list baseline against ML models and feature types (bag-of-words, TF-IDF, word embeddings). Comparative opinions ("X is better than Y") are out of scope.

Repo: `Evanwood5/Sentiment-Analysis`. Work goes in through branches plus PRs into `main`. Past branches: `evan`, `adiva`.

## Ground rules (from ReadMe.md and the team)
- Always load `data/processed/train.csv` and `data/processed/test.csv`. **Never modify or re-split them.**
- Train only on train and score only on test.
- Report accuracy, **macro-averaged** precision, recall and F1, plus a confusion matrix.
- Each model notebook saves `results_0X.csv` with exactly these columns: `model,accuracy,precision,recall,f1` (rounded to 3 decimals). `04_compare.ipynb` globs `results_*.csv`.
- Setup: `pip install -r requirements.txt` (pandas, scikit-learn, nltk, matplotlib, seaborn, gensim, ipykernel). Teammates use VS Code + Jupyter, Python 3.12.

## Data
- Source: Bing Liu's annotated customer reviews, in `data/raw/`: CustomerReviewData (5 products), Reviews-9-products, CustomerReviews-3-domains, and opinion-lexicon-English (2,006 positive and 4,783 negative words).
- The raw files have 10,992 lines. Labels are inline tags, e.g. `battery[-2]##sentence`. `00_data_prep.ipynb` sums the tag scores, drops sentences with no tag or a zero sum, lowercases, and removes 5 duplicates, leaving 4,574 sentences.
- Split: stratified 80/20 with `random_state=42`. Train has 3,659 rows and test has 915 (595 positive, 320 negative). About 65% positive and 35% negative.
- CSV columns: `text,label,product,source`. `label` is either "positive" or "negative".

## Notebooks
| File | Owner | Content |
|---|---|---|
| `00_data_prep.ipynb` | Evan | parse, clean, split (already run) |
| `01_wordlist.ipynb` | Adiva | lexicon count baseline: negative if neg > pos, else positive; a second version flips the 3 words after a negator (not, never, didn't, ...) |
| `02_naive_bayes.ipynb` | Evan | MultinomialNB with CountVectorizer and with TfidfVectorizer |
| `03_logreg_embeddings.ipynb` | Tanmay | LogisticRegression (max_iter=1000) with TF-IDF, TF-IDF + `class_weight="balanced"`, Word2Vec (trained on train, 100d, window 5, min_count 2, 20 epochs, seed 42, workers 1) and GloVe (`glove-wiki-gigaword-100` through gensim.downloader, 128 MB on first run). Sentence vector = mean of in-vocabulary word vectors; tokenizer `re.findall(r"[a-z']+", s)` |
| `04_compare.ipynb` | shared | concat results, sort by f1, bar chart; saves `figures/fig4_model_comparison.png` |

`03` also saves `figures/fig3_logreg_confusion.png`.

## Results (915-sentence test set, macro averages)
| Model | Acc | Prec | Rec | F1 | Negatives caught /320 |
|---|---|---|---|---|---|
| NB + Bag-of-Words | 0.826 | 0.819 | 0.790 | 0.801 | 214 (67%) |
| LR + TF-IDF (balanced) | 0.811 | 0.793 | 0.811 | 0.799 | 259 (81%) |
| LR + TF-IDF | 0.803 | 0.798 | 0.756 | 0.769 | 192 (60%) |
| LR + GloVe | 0.780 | 0.764 | 0.739 | 0.747 | 192 (60%) |
| Word list + negation | 0.760 | 0.774 | 0.681 | 0.693 | 134 (42%) |
| LR + Word2Vec | 0.736 | 0.714 | 0.677 | 0.685 | 154 (48%) |
| Word list | 0.729 | 0.721 | 0.649 | 0.656 | 123 (38%) |
| NB + TF-IDF | 0.742 | 0.828 | 0.636 | 0.633 | 90 (28%) |

Key findings:
- NB + BoW and balanced LR + TF-IDF are effectively tied for best; the balanced LR model catches the most negative sentences.
- TF-IDF hurts NB (it shrinks counts, so the model falls back on its positive prior) but works well with LR.
- Averaged embeddings underperform TF-IDF: averaging loses negation and word order, and Word2Vec has too little training data.
- Every model is biased toward the positive class.
- Word-list error types (220 errors, negation version): no opinion words (119), positive words in a complaint (67), negation the rule misses (34).

## Report (shared doc, outside the repo)
Sections: Goals, EDA, Methods, Results (1 word list, 2 NB, 3 LR), Comparison, Challenges.
- Sections 3 and Comparison are drafted in `writeup_sections.md`.
- Known fixes for the doc: in the NB section change "Figure X" to "Figure 2"; delete the leftover "Fill in the names and the bracketed words…" note; in the word-list section say that the confusion-matrix numbers (561/595 positive, 134/320 negative) are for the negation version.
- Figures: 1 = word-list confusion matrices, 2 = NB confusion matrices, 3 = LR confusion matrices, 4 = comparison chart.

## Status / open items
- Part 3 lives on branch `claude/brave-cerf-avuhcu`, **not merged to `main` yet**. Tanmay's collaborator invite expired, so a PR needs Evan to merge it or Tanmay needs a new invite.
- The balanced-LR model is an addition beyond the original plan; the team should confirm they want to keep it.
- The report doc still needs sections 3 and Comparison plus Figures 3 and 4 pasted in.

## Conventions
- Notebook style: short `#comment` header per cell, pandas + sklearn Pipelines, `ConfusionMatrixDisplay(cmap="Blues", colorbar=False)`, `classification_report(digits=3)`.
- Keep the report writing plain and student-level, and match the existing tone.
- Execute notebooks headless with: `jupyter nbconvert --to notebook --execute --inplace <file>.ipynb`
