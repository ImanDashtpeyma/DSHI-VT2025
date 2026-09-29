# Student Success Classification

A small data science project I did for the course Data Science for Health Informatics (VT2025). It uses the UCI student alcohol consumption dataset and tries to predict whether a student passes the final grade, mostly from study time, past failures and absences.

The main notebook is [student-success-classification.ipynb](student-success-classification.ipynb).

## Data

The data comes from [Kaggle / UCI](https://www.kaggle.com/datasets/uciml/student-alcohol-consumption/data) and covers students at two Portuguese secondary schools. There are two files, one for Maths (395 students) and one for Portuguese (649 students). I merged them and added a `course` column, which gives 1,044 rows and 34 columns. The merged file is `Data/combined_student_data.csv`, and `Test/app.py` is the small script that creates it.

## What I looked at first

Before any modelling I did some basic exploration. A few of the results:

- 211 of the 1,044 students report high weekend drinking (`Walc > 3`).
- The mean final grade (`G3`) is 11.34.
- Boys drink more than girls, on weekdays (1.78 vs 1.27) and on weekends (2.73 vs 1.94). Girls have a slightly higher mean grade (11.45 vs 11.20).
- Mean grade goes down as weekend drinking goes up, from 11.74 at level 1 to 10.40 at level 5. Absences go up too, from 3.6 to 6.0 between levels 1 and 4.
- Study time has a weak positive correlation with the final grade (0.16) and a weak negative one with drinking (-0.16 weekdays, -0.23 weekends).

These are only correlations, so I don't read anything causal into them.

## Classification

The target is `passed`, meaning `G3 >= 10`. About 78% of the students pass. I used `studytime`, `failures`, `absences`, `school` and `sex` as features, with the two categorical ones one-hot encoded. The data is split 80/20 with stratification. I compared ten models (decision tree, random forest, KNN, SVM and logistic regression, two settings each) with 5-fold cross-validation on the training part. KNN, SVM and logistic regression get a `StandardScaler` inside a pipeline, and I added a "always predict pass" baseline for reference.

Cross-validation results, sorted by F1:

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| LR_1 | 0.8036 | 0.8170 | 0.9647 | 0.8845 |
| LR_2 | 0.8024 | 0.8142 | 0.9678 | 0.8842 |
| SVM_1 (linear) | 0.7988 | 0.8119 | 0.9662 | 0.8822 |
| SVM_2 (RBF) | 0.8024 | 0.8302 | 0.9385 | 0.8810 |
| RF_2 | 0.8000 | 0.8244 | 0.9447 | 0.8803 |
| Baseline | 0.7796 | 0.7796 | 1.0000 | 0.8762 |
| KNN_2 | 0.7892 | 0.8286 | 0.9201 | 0.8719 |
| DT_1 | 0.7892 | 0.8294 | 0.9185 | 0.8715 |
| DT_2 | 0.7868 | 0.8348 | 0.9062 | 0.8689 |
| RF_1 | 0.7772 | 0.8235 | 0.9093 | 0.8642 |
| KNN_1 | 0.7725 | 0.8244 | 0.9001 | 0.8605 |

Logistic regression (LR_1) came out on top and is also very cheap to train, so I picked it. On the test set it gets 0.799 accuracy and 0.882 F1, while the baseline gets 0.780 and 0.876. So it is only slightly better than always guessing "pass". It found just 10 of the 46 students who actually failed.

Scaling did not help on this data, and the gaps between the top models are small enough to be noise.

## Trying to find the students who fail

Since accuracy hides the problem, I looked at the failed class on its own and tried two things: `class_weight='balanced'`, and adding the earlier grades `G1` and `G2` as features. Test set results (209 students, 46 failed):

| Model | Features | Precision (failed) | Recall (failed) | ROC-AUC | Failed students found |
|---|---|---|---|---|---|
| Baseline | 5 features | 0.00 | 0.00 | 0.50 | 0 / 46 |
| Logistic regression | 5 features | 0.63 | 0.22 | 0.70 | 10 / 46 |
| Logistic regression, balanced | 5 features | 0.45 | 0.41 | 0.70 | 19 / 46 |
| Random forest, balanced | + `G1`, `G2` | 0.70 | 0.89 | 0.96 | 41 / 46 |

Class weighting roughly doubles recall but costs precision, and the model with only the original features is still weak. With `G1` and `G2` the results are much better, but `G2` is close to the final grade, so that mostly tells us that students who are already failing halfway keep failing. It doesn't show we can predict failure early from behaviour. With only 46 failed students in the test set the numbers also move around a lot, so I trust the cross-validation table in the notebook more.

## Things I would do next

Tune the decision threshold, use repeated cross-validation to get tighter estimates, and try a model that is easier to explain (coefficients or feature importances), since that would matter if this were used to decide who gets support.

## Running it

```bash
git clone https://github.com/ImanDashtpeyma/DSHI-VT2025.git
cd DSHI-VT2025
pip install -r packages.txt
jupyter notebook student-success-classification.ipynb
```

The main libraries are pandas, numpy, scikit-learn, matplotlib and seaborn.

## Repository layout

```
student-success-classification.ipynb   main project
coursework/                            earlier homework and labs (EDA, clustering)
Data/                                  student and bank marketing CSV files
Test/app.py                            merges the two student files
packages.txt                           Python dependencies
```
