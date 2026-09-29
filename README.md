# 🎓 Predicting Student Success: Alcohol Consumption & Academic Performance

> Exploratory data analysis and machine-learning classification on the UCI *Student Alcohol Consumption* dataset: can we predict whether a student passes the final exam from study habits, absences, and background?

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-EDA-150458?logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

**Author:** Iman Dashtpeyma · Course: *Data Science for Health Informatics / Design* (VT2025)
**Main notebook:** [`student-success-classification.ipynb`](student-success-classification.ipynb)

---

## 📌 Overview

The project has two parts:

1. **Exploratory Data Analysis (EDA)**: how do alcohol consumption, gender, study time and absences relate to students' final grades?
2. **Classification**: a pass/fail model (`G3 >= 10`) trained and compared across **5 algorithms × 2 hyperparameter settings = 10 models**, using stratified 5-fold cross-validation.

## 📊 Dataset

| | |
|---|---|
| **Source** | [UCI / Kaggle: Student Alcohol Consumption](https://www.kaggle.com/datasets/uciml/student-alcohol-consumption/data) |
| **Origin** | School reports from two Portuguese secondary schools |
| **Files** | `student-mat.csv` (395 students, Maths) + `student-por.csv` (649 students, Portuguese) |
| **Combined** | **1,044 rows × 34 columns** (33 original + a `course` column I added) |
| **Features** | Demographic, social, family and school-related variables (numeric, ordinal, nominal, binary) |

The two files are merged into `Data/combined_student_data.csv` (see also [`Test/app.py`](Test/app.py)).

## 🔍 Key EDA Findings

| Question | Result |
|---|---|
| Students with high weekend drinking (`Walc > 3`) | **211** of 1,044 |
| Mean final grade (`G3`) | **11.34** |
| Alcohol use by gender (Dalc / Walc) | Male **1.78 / 2.73** vs. Female **1.27 / 1.94** |
| Mean final grade by gender | Female **11.45** vs. Male **11.20** |
| Grade vs. weekend alcohol (`Walc` 1 → 5) | **11.74 → 10.40** (declining trend) |
| Absences vs. weekend alcohol (`Walc` 1 → 4) | **3.6 → 6.0** average absences |
| Correlation: study time ↔ `G3` | **+0.16** (weak positive) |
| Correlation: study time ↔ `Dalc` / `Walc` | **−0.16 / −0.23** (weak negative) |

**Takeaway:** heavier drinking is associated with lower grades and more absences, and students who study more tend to drink less. These are *correlations*, not causal claims.

## 🤖 Machine Learning Pipeline

**Target:** `passed = (G3 >= 10)` (about 78% of students pass, so the classes are imbalanced)
**Features:** `studytime`, `failures`, `absences` (numeric) + `school`, `sex` (one-hot encoded)
**Split:** 80/20 stratified train/test (`random_state=42`); models compared with stratified 5-fold CV on the training set (835 rows), plus a `DummyClassifier` majority-class **baseline**
**Scaling:** `StandardScaler` inside a scikit-learn `Pipeline` for KNN, SVM and Logistic Regression (re-fit per CV fold, so no leakage); tree-based models are left unscaled

| Algorithm | Variant 1 | Variant 2 |
|---|---|---|
| Decision Tree | `max_depth=3` | `max_depth=10, min_samples_split=10` |
| Random Forest | 100 trees | 50 trees, `max_depth=5` |
| K-Nearest Neighbors | `k=3` | `k=7` |
| SVM | linear, `C=1` | RBF, `C=10` |
| Logistic Regression | `C=1` | `C=0.1` (L2) |

### Results (5-fold CV, sorted by F1)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **LR_1** | 0.8036 | 0.8170 | 0.9647 | **0.8845** |
| LR_2 | 0.8024 | 0.8142 | 0.9678 | 0.8842 |
| SVM_1 (linear) | 0.7988 | 0.8119 | 0.9662 | 0.8822 |
| SVM_2 (RBF) | 0.8024 | 0.8302 | 0.9385 | 0.8810 |
| RF_2 | 0.8000 | 0.8244 | 0.9447 | 0.8803 |
| *Baseline (always "pass")* | *0.7796* | *0.7796* | *1.0000* | *0.8762* |
| KNN_2 | 0.7892 | 0.8286 | 0.9201 | 0.8719 |
| DT_1 | 0.7892 | 0.8294 | 0.9185 | 0.8715 |
| DT_2 | 0.7868 | 0.8348 | 0.9062 | 0.8689 |
| RF_1 | 0.7772 | 0.8235 | 0.9093 | 0.8642 |
| KNN_1 | 0.7725 | 0.8244 | 0.9001 | 0.8605 |

### Held-out test set (209 students, evaluated once)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Baseline (always "pass") | 0.7799 | 0.7799 | 1.0000 | 0.8763 |
| **LR_1** (recommended) | 0.7990 | 0.8135 | 0.9632 | 0.8820 |

`LR_1` confusion matrix `[[TN FP] [FN TP]]` = `[[10, 36], [6, 157]]`: it catches **10 of the 46** students who actually failed.

**Conclusion (Scenario 1):** scaled Logistic Regression had the best CV F1 and is the cheapest good model, so it is my recommendation. But the honest takeaway is that no model is meaningfully better than the naive baseline with these five features, which motivated Scenario 2 below.

### 🎯 Scenario 2: catching failing students

Because accuracy/F1 on the majority class is misleading, I re-evaluated with metrics on the **failed** class, then tested two changes: **class weighting** (`class_weight='balanced'`) and adding the earlier grades **`G1`, `G2`**. Test-set results (209 students, 46 of whom failed):

| Model | Features | Failed-class precision | Failed-class recall | ROC-AUC | Failing students caught |
|---|---|---|---|---|---|
| Baseline (always "pass") | 5 features | 0.00 | 0.00 | 0.50 | 0 / 46 |
| Logistic Regression | 5 features | 0.63 | 0.22 | 0.70 | 10 / 46 |
| Logistic Regression, **balanced** | 5 features | 0.45 | 0.41 | 0.70 | 19 / 46 |
| Random Forest, **balanced** | + `G1`, `G2` | 0.70 | 0.89 | 0.96 | **41 / 46** |

- Class weighting roughly **doubles recall** on failing students but lowers precision: a trade-off, and the 5-feature model stays weak (ROC-AUC ≈ 0.70).
- Adding `G1`/`G2` makes the task far easier (CV ROC-AUC ≈ 0.97). Caveat: `G2` is close to the final grade, so this shows late-stage identification, not early prediction from behaviour or background.
- With only 46 failing students in the test set these numbers are noisy; the CV results in the notebook rank the configurations the same way.

### ⚠️ Limitations & honest notes

- **Scenario 1 is barely above baseline.** About 78% of students pass, so "everyone passes" already scores ~78% accuracy and 0.876 F1. The best model reaches ~80% accuracy / 0.882 F1 on the test set, and most models have *lower* CV F1 than the baseline.
- **Differences between the top Scenario-1 models are within noise** (≈0.004 F1 on ~835 training rows), so the ranking should not be over-interpreted.
- **Scaling was applied** (`StandardScaler` in a pipeline) but did not improve results on this data.
- **The `G1`/`G2` scenario is a different question** ("who will fail, given mid-year grades?"), and its strong results should not be read as early-warning performance.
- **Next steps:** threshold tuning / probability calibration, stratified repeated CV for tighter confidence intervals, and an interpretable model (coefficients / feature importances) for the intervention use case.

## 🚀 Getting Started

```bash
git clone https://github.com/ImanDashtpeyma/DSHI-VT2025.git
cd DSHI-VT2025
pip install -r packages.txt
jupyter notebook student-success-classification.ipynb
```

Core libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`.

## 📁 Repository Structure

```
DSHI-VT2025/
├── student-success-classification.ipynb   # ⭐ Main project: EDA + classification
├── coursework/                      # Earlier course assignments & labs
│   ├── DSHI_HW1_Iman_Dashtpeyma.ipynb   # EDA on the same dataset
│   ├── DSHI_HW2_Iman_Dashtpeyma.ipynb   # Clustering: K-Means & DBSCAN on synthetic data
│   ├── Lab2-ExploratoryDataAnalysis_EDA.ipynb
│   └── Lab3-Clustering_UnsupervisedLearning.ipynb
├── Data/                            # Student and bank-marketing CSVs
├── Test/app.py                      # Script that merges the two student datasets
└── packages.txt                     # Python dependencies
```

## 🛠️ Skills Demonstrated

`Exploratory data analysis` · `Feature encoding` · `Stratified splitting` · `Class-imbalance handling (class weighting)` · `Cross-validated model comparison` · `Classification (DT, RF, KNN, SVM, LR)` · `Result interpretation`

---

*Iman Dashtpeyma · Software developer with a background in network security and DevOps, now building data science skills.*
