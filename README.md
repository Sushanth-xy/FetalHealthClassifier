## Multiclass Classification of Fetal Health using Cardiotocogram (CTG) Data

Machine Learning, Mini-Project

**Team:** [TANISHA DALMIA, PES2UG24CS550] and [SUSHANTH B, PES2UG24CS543]  
**Section:** [I]  **Problem No.:** [44]

## Problem

Classify a fetus as **Normal**, **Suspect** or **Pathological** from 21 numeric features extracted from a cardiotocogram, and compare 10 machine-learning models.

## Dataset

Cardiotocography dataset (Ayres de Campos et al., 2000), UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/193/cardiotocography

- 2,126 raw records, 21 features, 3 classes
- After dropping nulls and duplicates: 2,113 samples (1,646 Normal / 292 Suspect / 175 Pathological)
- Place the CSV at `data/fetal_health.csv` (the label column is `fetal_health`, with values 1, 2, 3)

## Repository structure

```
.
├── README.md
├── requirements.txt
├── data/
│   └── fetal_health.csv
├── notebooks/
│   └── fetal_health.ipynb      # full walkthrough with plots
├── src/
│   └── train.py                # runs the whole pipeline from the command line
├── results/                    # saved tables and figures
└── Mini_Project_Writeup.pdf
```

> Keep this tree in sync with your actual repository. Delete any line for a file you do not have.

## Setup

Requires Python 3.9 or newer.

```bash
git clone <your-repo-url>
cd <your-repo-name>

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

`requirements.txt`:

```
numpy
pandas
scikit-learn
xgboost
lightgbm
matplotlib
seaborn
jupyter
```

## How to run

**Option A: notebook (recommended for the demo)**

```bash
jupyter notebook notebooks/fetal_health.ipynb
```

Then choose *Kernel, Restart & Run All*.

**Option B: script**

```bash
python src/train.py --data data/fetal_health.csv --test-size 0.3 --seed 42
```

## What the pipeline does

1. **Load and clean:** read the CSV, drop null and duplicate rows, print the class distribution.
2. **Split:** stratified 70/30 train/test split with a fixed random seed.
3. **Scale and select features:** standardise features and keep the top 10 by SelectKBest, fit on the training set only to avoid data leakage.
4. **Train 10 models:** Logistic Regression, Linear SVC, SVC (RBF), KNN, Decision Tree, Random Forest, MLP, Gradient Boosting, XGBoost, LightGBM.
5. **Evaluate:** test accuracy for every model, then a classification report (precision, recall, F1) and confusion matrix for the best model.
6. **Tune:** grid search with 3-fold cross-validation on LightGBM, then compare before and after tuning.

## Results

Numbers reported by the original paper (single 70/30 split). Fill in the last column with your own run.

| Model               | Original paper | This reproduction |
|---------------------|:--------------:|:-----------------:|
| Gradient Boosting   | 0.9558 | [fill] |
| LightGBM            | 0.9558 | [fill] |
| Random Forest       | 0.9495 | [fill] |
| XGBoost             | 0.9495 | [fill] |
| KNN                 | 0.9401 | [fill] |
| MLP                 | 0.9353 | [fill] |
| Decision Tree       | 0.9338 | [fill] |
| SVC (RBF)           | 0.9211 | [fill] |
| Logistic Regression | 0.8959 | [fill] |
| Linear SVC          | 0.8927 | [fill] |

Your numbers may differ slightly from the paper because of the random seed, library versions, and because feature selection here is fit on the training data only.

## Reproducibility notes

- Random seed is fixed (`42`) for the split and for all models that accept one.
- Tested with Python [x.y] and the library versions in `requirements.txt` (run `pip freeze` and pin them before submitting).

## Contributions

- [Name 1]: [what you did]
- [Name 2]: [what you did]

## References

- Ayres de Campos et al. (2000). SisPorto 2.0: A Program for Automated Analysis of Cardiotocograms. J Matern Fetal Med 5:311-318.
- Ke, G. et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. NeurIPS 30.
- Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12.
