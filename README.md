# Multiclass Classification of Fetal Health using Cardiotocogram (CTG) Data

Machine Learning Mini-Project

| | |
|---|---|
| **Team** | Tanisha Dalmia (PES2UG24CS550), Sushanth B (PES2UG24CS543) |
| **Section** | I |
| **Problem No.** | 44 |

## Problem

Classify a fetus as **Normal**, **Suspect** or **Pathological** from 21 numeric features extracted from a cardiotocogram (CTG), and compare several machine-learning models.

## Dataset

Cardiotocography dataset (Ayres de Campos et al., 2000), originally from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/193/cardiotocography). The notebook loads the Kaggle copy, [`andrewmvd/fetal-health-classification`](https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification), automatically through `kagglehub`, so there is nothing to download by hand.

- 2,126 records, 21 features, 1 target column (`fetal_health`)
- Classes: `1` = Normal, `2` = Suspect, `3` = Pathological
- Imbalanced: in the 426-row test split, 332 are Normal, 59 Suspect and 35 Pathological

## Repository structure

```
.
├── README.md
├── requirements.txt
├── main.ipynb                  # full pipeline: load, train, evaluate, tune, save
├── test.ipynb                  # tests the final saved model
└── xgb_fetal_health.joblib     # final tuned XGBoost model (saved by main.ipynb)
```

## Setup

Requires Python 3.9 or newer (developed on Python 3.13).

```bash
git clone https://github.com/Sushanth-xy/FetalHealthClassifier.git
cd FetalHealthClassifier

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

`kagglehub` may ask you to authenticate with Kaggle the first time it downloads the dataset. See the [kagglehub docs](https://github.com/Kaggle/kagglehub) for setting up credentials. On Google Colab it works without extra setup.

## How to run

```bash
jupyter notebook main.ipynb
```

Then choose *Kernel, Restart & Run All*.

To try the final saved model, open `test.ipynb`. It loads `xgb_fetal_health.joblib`, so it needs no retraining.

The grid search (480 fits) and the stacking comparison are the slow cells and can take several minutes.

## What the pipeline does

1. **Load:** download `fetal_health.csv` through `kagglehub` and preview it.
2. **Split:** stratified 80/20 train/test split (`random_state=42`).
3. **Single-split baselines:** train and evaluate on the test set, with accuracy, a classification report and a confusion matrix for each:
   Gradient Boosting, Random Forest, KNN (k=5), LightGBM, XGBoost, Decision Tree and SVC (RBF).
   KNN and SVC use features standardised with a `StandardScaler` fit on the training set only. KNN additionally uses `SelectKBest(f_classif, k=10)`, also fit on the training set only, which raised its test accuracy from 0.8709 to 0.9108.
4. **Cross-validated comparison (training set only):** 5-fold stratified CV repeated 3 times, reporting accuracy, macro F1 and per-class recall for each model. A second run adds Soft Voting and Stacking ensembles (Random Forest + LightGBM + XGBoost + SVC).
5. **Class imbalance:** XGBoost retrained with balanced sample weights to improve recall on the Suspect class.
6. **Tune:** grid search (96 parameter combinations, 5-fold CV, scored on macro F1) on the balanced-weight XGBoost.
7. **Save:** the tuned XGBoost model is fit on the training split and saved with `joblib` as `xgb_fetal_health.joblib`.

## Results

### Single 80/20 hold-out split (426 test samples)

| Model               | Test accuracy | Macro F1 |
|---------------------|:-------------:|:--------:|
| LightGBM            | 0.9437 | 0.89 |
| XGBoost             | 0.9390 | 0.89 |
| Gradient Boosting   | 0.9296 | 0.87 |
| Random Forest       | 0.9249 | 0.85 |
| KNN (k=5, SelectKBest 10) | 0.9108 | 0.82 |
| Decision Tree       | 0.9014 | 0.82 |
| SVC (RBF)           | 0.8944 | 0.78 |

### Repeated 5-fold cross-validation (training set, mean ± std)

| Model                   | Accuracy | Macro F1 | Recall: Normal | Recall: Suspect | Recall: Pathological |
|-------------------------|:--------:|:--------:|:--------------:|:---------------:|:--------------------:|
| XGBoost                 | 0.955 ± 0.007 | 0.920 ± 0.015 | 0.983 | 0.804 | 0.936 |
| Stacking                | 0.953 ± 0.006 | 0.920 ± 0.012 | 0.981 | 0.807 | 0.941 |
| Soft Voting             | 0.953 ± 0.005 | 0.918 ± 0.012 | 0.983 | 0.795 | 0.936 |
| LightGBM                | 0.954 ± 0.005 | 0.920 ± 0.011 | 0.984 | 0.804 | 0.931 |
| LightGBM (balanced class weight) | 0.953 ± 0.006 | 0.918 ± 0.009 | 0.977 | 0.835 | 0.934 |
| Gradient Boosting       | 0.953 ± 0.007 | 0.917 ± 0.014 | 0.982 | 0.815 | 0.912 |
| Random Forest           | 0.942 ± 0.005 | 0.894 ± 0.017 | 0.981 | 0.763 | 0.877 |
| Decision Tree           | 0.922 ± 0.017 | 0.874 ± 0.025 | 0.956 | 0.736 | 0.917 |
| SVC (RBF, scaled)       | 0.910 ± 0.014 | 0.831 ± 0.026 | 0.972 | 0.650 | 0.766 |
| KNN (scaled)            | 0.904 ± 0.014 | 0.820 ± 0.025 | 0.970 | 0.629 | 0.749 |
| XGBoost balanced        | 0.952 ± 0.007 | 0.919 ± 0.010 | 0.972 | 0.842 | 0.953 |
| XGBoost balanced + grid | 0.951 ± 0.006 | 0.919 ± 0.010 | 0.965 | 0.871 | 0.953 |

### Findings

- Gradient-boosted trees (XGBoost, LightGBM, Gradient Boosting) clearly beat the distance- and kernel-based models.
- The Suspect class is the hardest to detect, which is expected given how few Suspect samples there are.
- Balanced sample weights plus grid search raised Suspect recall from 0.804 to 0.871 in cross-validation. Overall accuracy and macro F1 stayed essentially flat (about 0.95 and 0.92), so the gain comes from trading a little Normal-class recall for better detection of Suspect cases. Whether that trade is worth it depends on the cost of a missed Suspect case.
- Best grid-search parameters: `max_depth=7`, `learning_rate=0.03`, `n_estimators=200`, `subsample=0.8`, `colsample_bytree=1.0`, `min_child_weight=1`.

### Comparison with the original paper

The paper reported the following accuracies using a single 70/30 split on the UCI data. Our setup differs (80/20 split, Kaggle copy, no feature selection), so the numbers are not directly comparable.

| Model             | Original paper | This project (80/20 split) |
|-------------------|:--------------:|:--------------------------:|
| Gradient Boosting | 0.9558 | 0.9296 |
| LightGBM          | 0.9558 | 0.9437 |
| Random Forest     | 0.9495 | 0.9249 |
| XGBoost           | 0.9495 | 0.9390 |
| KNN               | 0.9401 | 0.9108 |
| Decision Tree     | 0.9338 | 0.9014 |
| SVC (RBF)         | 0.9211 | 0.8944 |

The paper's Logistic Regression, Linear SVC and MLP were not run in this project.

## Reproducibility notes

- Random seed is fixed (`42`) for the split, the CV folds and every model that accepts one.
- Results can vary slightly with library versions and across machines (especially multi-threaded boosting).
- Tested on Google Colab with Python 3.13. Run `pip freeze > requirements-lock.txt` if you need exact pins.

## Limitations

- In `main.ipynb`, the tuned XGBoost model is evaluated with cross-validation on the training split only. Final held-out results for the saved model are in `test.ipynb`.
- The saved `xgb_fetal_health.joblib` is trained on the 80% training split only, not the full dataset.
- Labels are 0-indexed (`y - 1`) for XGBoost, so the saved model predicts `0/1/2`, not `1/2/3`. Add 1 to map back to Normal/Suspect/Pathological.
- This is a student project and not a clinical tool.

## Contributions

- **Sushanth:** report, code, testing
- **Tanisha:** report, presentation

## References

- Ayres de Campos, D. et al. (2000). SisPorto 2.0: A Program for Automated Analysis of Cardiotocograms. *J Matern Fetal Med* 5:311-318.
- Chen, T. and Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD '16*.
- Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS 30*.
- Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR* 12.
