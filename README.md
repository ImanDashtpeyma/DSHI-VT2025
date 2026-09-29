# 🎓 Predicting Student Success: Alcohol Consumption & Academic Performance

> Exploratory data analysis and machine-learning classification on the UCI *Student Alcohol Consumption* dataset: can we predict whether a student passes the final exam from study habits, absences, and background?

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-EDA-150458?logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

**Author:** Iman Dashtpeyma · Course: *Data Science for Health Informatics / Design* (VT2025)
**Main notebook:** [`DSHI_HW3_Iman_Dashtpeyma.ipynb`](DSHI_HW3_Iman_Dashtpeyma.ipynb)

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
**Split:** 80/20 stratified train/test (`random_state=42`); models compared with 5-fold CV on the training set

| Algorithm | Variant 1 | Variant 2 |
|---|---|---|
| Decision Tree | `max_depth=3` | `max_depth=10, min_samples_split=10` |
| Random Forest | 100 trees | 50 trees, `max_depth=5` |
| K-Nearest Neighbors | `k=3` | `k=7` |
| SVM | linear, `C=1` | RBF, `C=10` |
| Logistic Regression | `C=1` | `C=0.1` (L2) |

### Results (5-fold CV, sorted by F1)

| Model | Accuracy | Precision | Recall | F1 | Fit time (s) |
|---|---|---|---|---|---|
| **SVM_2** (RBF) | 0.8096 | 0.8284 | 0.9539 | **0.8864** | 0.0126 |
| **LR_1** | 0.8048 | 0.8181 | 0.9647 | 0.8851 | 0.0041 |
| KNN_2 | 0.8060 | 0.8238 | 0.9554 | 0.8847 | 0.0012 |
| SVM_1 (linear) | 0.8024 | 0.8153 | 0.9662 | 0.8839 | 0.0046 |
| LR_2 | 0.7976 | 0.8099 | 0.9677 | 0.8817 | 0.0031 |
| DT_1 | 0.7976 | 0.8270 | 0.9370 | 0.8782 | 0.0010 |
| RF_2 | 0.7928 | 0.8208 | 0.9400 | 0.8761 | 0.0368 |
| DT_2 | 0.7916 | 0.8374 | 0.9093 | 0.8716 | 0.0012 |
| KNN_1 | 0.7749 | 0.8225 | 0.9079 | 0.8626 | 0.0010 |
| RF_1 | 0.7749 | 0.8232 | 0.9063 | 0.8624 | 0.0790 |

**Conclusion:** SVM with an RBF kernel scored best on F1, but Logistic Regression (`LR_1`) is within 0.0013 F1 at roughly a third of the training time, making it the more practical choice for deployment.

### ⚠️ Limitations & honest notes

- **Small margin over a naive baseline.** About 78% of students pass, so a model that always predicts "pass" already reaches ~78% accuracy. The best models (~81%) improve only modestly, which is expected given only five simple features.
- `G1` and `G2` (earlier grades) were intentionally **not** used as features; they would make prediction much easier but far less useful for early intervention.
- Results are cross-validation scores on the training split; the held-out test set is reserved for a final evaluation of the chosen model.
- Distance-based models (KNN, SVM) would likely benefit from feature scaling.

## 🚀 Getting Started

```bash
git clone https://github.com/ImanDashtpeyma/DSHI-VT2025.git
cd DSHI-VT2025
pip install -r packages.txt
jupyter notebook DSHI_HW3_Iman_Dashtpeyma.ipynb
```

Core libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`.

## 📁 Repository Structure

```
DSHI-VT2025/
├── DSHI_HW3_Iman_Dashtpeyma.ipynb   # ⭐ Main project: EDA + classification
├── DSHI_HW1_Iman_Dashtpeyma.ipynb   # EDA on the same dataset
├── DSHI_HW2_Iman_Dashtpeyma.ipynb   # Clustering: K-Means & DBSCAN on synthetic data
├── Lab2-ExploratoryDataAnalysis_EDA.ipynb
├── Lab3-Clustering_UnsupervisedLearning.ipynb
├── Data/                            # Student and bank-marketing CSVs
├── Test/app.py                      # Script that merges the two student datasets
└── packages.txt                     # Python dependencies
```

## 🛠️ Skills Demonstrated

`Exploratory data analysis` · `Feature encoding` · `Stratified splitting` · `Class-imbalance awareness` · `Cross-validated model comparison` · `Classification (DT, RF, KNN, SVM, LR)` · `Result interpretation`

---

*Iman Dashtpeyma · Software developer with a background in network security and DevOps, now building data science skills.*
