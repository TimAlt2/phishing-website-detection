# Phishing Website Detection
A comparative machine-learning experiment for detecting phishing websites using four classifiers:
Logistic Regression, Decision Tree, Random Forest, and Support Vector Machine (SVM).

## Dataset
The experiment uses the **Phishing Websites** dataset from the UCI Machine Learning Repository
(dataset ID 327): https://archive.ics.uci.edu/dataset/327/phishing

- 11,055 records
- 30 input features
- 6,157 legitimate records (55.69%)
- 4,898 phishing records (44.31%)
- No missing values reported by the dataset analysis

The original target labels are mapped as follows:
- `-1` → phishing (`1`)
- `1` → legitimate (`0`)

The positive class used for Precision, Recall, F1-score, ROC-AUC and Average Precision is **Phishing**.

## Methodology
1. Download the dataset through `ucimlrepo`.
2. Hash identical feature rows and use the hashes as group identifiers.
3. Create a held-out test partition using the first fold of a 5-fold `StratifiedGroupKFold`.
4. Tune each model using `GridSearchCV` with a separate 5-fold `StratifiedGroupKFold` applied only to the training partition.
5. Optimize the F1-score during hyperparameter selection.
6. Evaluate the selected estimator once on the held-out test partition.

Grouping identical feature rows is intended to prevent matching feature rows from appearing in both training and test partitions. The script checks for overlap and stops if any shared groups are found.

**Important limitation:** the current experiment uses one held-out outer fold as the test set. It does not report repeated outer cross-validation or external-dataset validation. The dataset analysis also identifies groups in which identical feature rows have conflicting target labels; these records are retained and grouped together.

## Models and tuning
- **Logistic Regression:** `C`, `solver`
- **Decision Tree:** `criterion`, `max_depth`, `min_samples_split`, `min_samples_leaf`
- **Random Forest:** `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`
- **SVM (RBF kernel):** `C`, `gamma`

The exact search grids, fold-level scores, selected parameters and cross-validation summaries are stored in `results/tables/`.

## Evaluation
The test evaluation reports:
- Accuracy
- Precision
- Recall
- F1-score
- Specificity
- ROC-AUC
- Average Precision

The repository includes the test metrics, confusion matrices, ROC curves, Precision–Recall curves,
model comparison chart and class distribution chart in `results/`.

## Project structure
```text
phishing-website-detection/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   └── phishing_detection.py
└── results/
    ├── figures/
    └── tables/
```

## Installation
Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Activate the environment, then install dependencies:

```bash
pip install -r requirements.txt
```

## Run
```bash
python src/phishing_detection.py
```

The script downloads the dataset from UCI and writes fresh experiment outputs to
`phishing_results_tuned/` in the current working directory.

## Reproducibility notes
- Random seed: `42`
- Outer split: 5-fold `StratifiedGroupKFold`, first fold held out for testing
- Inner tuning: 5-fold `StratifiedGroupKFold`
- Hyperparameter selection metric: F1-score
- Test metrics are calculated only after tuning is complete.

## Data and code availability
Dataset: UCI Machine Learning Repository, *Phishing Websites*, dataset ID 327,
https://archive.ics.uci.edu/dataset/327/phishing

Code: this GitHub repository (insert the final repository URL after publication).

## Citation
If you use the dataset, cite its UCI repository record:

Mohammad, R. M., & McCluskey, L. (2015). *Phishing Websites* [Dataset].
UCI Machine Learning Repository. https://doi.org/10.24432/C51W2X
