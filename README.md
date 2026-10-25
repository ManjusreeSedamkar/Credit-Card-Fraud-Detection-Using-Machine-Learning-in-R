# Credit Card Fraud Detection Using Machine Learning in R

## Overview

This project implements a machine learning approach to detect fraudulent credit card transactions using R. It focuses on **imbalanced classification**, since fraudulent transactions represent only a very small proportion of all transactions.

The project explores the dataset, preprocesses transaction data, compares techniques for handling class imbalance, and evaluates four machine learning classifiers using metrics suitable for rare-event detection.

## Objectives

- Explore and preprocess credit card transaction data using R.
- Investigate class imbalance and transaction feature distributions.
- Compare class weighting, random undersampling, and SMOTE.
- Train and evaluate four machine learning models.
- Select the best-performing model using Precision, Recall, F1-score, ROC-AUC, and PR-AUC rather than accuracy alone.

## Dataset

The dataset contains anonymized credit card transactions made by European cardholders over two days.

- **Original transactions:** 284,807
- **Columns:** 31
- **Missing values:** 0
- **Duplicate rows removed:** 1,081
- **Transactions after cleaning:** 283,726
- **Legitimate transactions after cleaning:** 283,253
- **Fraudulent transactions after cleaning:** 473
- **Fraud rate after cleaning:** approximately 0.167%

The features include `Time`, anonymized PCA components `V1`–`V28`, `Amount`, and the target variable `Class` (0 = legitimate, 1 = fraudulent).

**Dataset source:** [Credit Card Fraud Detection Dataset — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## Methodology

### 1. Data cleaning and exploration

- Loaded the transaction data into R.
- Checked for missing values and duplicate rows.
- Examined class distribution, transaction amounts, transaction times, and feature correlations.
- Visualized key feature distributions by transaction class.

### 2. Preprocessing

- Split the cleaned dataset into stratified training (70%) and test (30%) sets.
- Standardized `Time` and `Amount` using the training-set mean and standard deviation.
- Kept the test set at its natural class distribution for realistic evaluation.

### 3. Handling class imbalance

Three strategies were compared:

- **Class weights:** Assigned greater importance to fraudulent transactions during model training.
- **Undersampling:** Randomly selected legitimate transactions to balance the training classes.
- **SMOTE:** Generated synthetic minority-class examples using nearest neighbours.

Resampling was applied to the training data only.

### 4. Machine learning models

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

### 5. Evaluation

The models were evaluated using:

- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Confusion matrix
- Five-fold stratified cross-validation

Accuracy was not used as the main selection criterion because the dataset is highly imbalanced.

## Results

The reported best-performing configuration was **XGBoost with class weights**.

| Metric | Test result |
|---|---:|
| Precision | 0.8626 |
| Recall | 0.8014 |
| F1-score | 0.8309 |
| ROC-AUC | 0.9756 |
| PR-AUC | 0.8268 |
| Balanced accuracy | 0.901 |

On the test set of 85,116 transactions, the model identified 113 of 141 fraudulent transactions, missed 28 frauds, and generated 18 false-positive alerts.

Random Forest with class weights was a close alternative, with a reported precision of 0.9217 and PR-AUC of 0.7997.

The results indicate that the imbalance-handling strategy has a substantial effect on model performance. Although undersampling and SMOTE increased recall in some experiments, they also produced considerably more false-positive alerts.

## Conclusion

XGBoost with class weights achieved the strongest overall balance between precision and recall among the evaluated configurations, with an F1-score of 0.8309 and PR-AUC of 0.8268. The experiment demonstrates why precision-recall metrics are more informative than accuracy for fraud detection when fraudulent transactions are rare.

The results also show that selecting an appropriate imbalance-handling technique is as important as choosing the classifier. Further work could include hyperparameter tuning, decision-threshold optimization, cost-sensitive learning, additional sampling methods, and deployment as a transaction-scoring application.

## Limitations and Future Work

- Evaluation used a single stratified train-test split.
- The test set contained only 141 fraudulent transactions.
- The classification threshold was fixed at 0.5.
- Cross-validation was conducted using undersampling, so its results do not directly validate the class-weighted configuration.
- Future work could tune model parameters and decision thresholds, evaluate additional imbalance-handling methods, and develop a simple scoring application.

## Technologies and Libraries

- R
- `dplyr`
- `ggplot2`
- `caret`
- `rpart`
- `ranger`
- `xgboost`
- `smotefamily`
- `pROC`
- `PRROC`

## Project Structure

```text
credit-card-fraud-detection-r/
├── case_study.R
├── case_study.Rmd
├── report.pdf
├── README.md
└── outputs/
    ├── 01_class_distribution.png
    ├── 02_amount_by_class.png
    ├── 03_time_density.png
    ├── 04_correlation_heatmap.png
    ├── 05_key_feature_densities.png
    ├── 06_model_comparison.png
    ├── 07_precision_recall_curves.png
    └── 08_xgboost_feature_importance.png
```

*Note: The filenames above are illustrative. Use the actual filenames generated for your project.*

## How to Run

1. Install R and RStudio.
2. Download the dataset from the source linked above, subject to its access and usage terms.
3. Place the dataset where the R script or `.Rmd` expects it, and update the file path if necessary.
4. Install the required R packages if they are not already installed.
5. Open `case_study.R` in RStudio and run the script, or open `case_study.Rmd` and use **Knit** to generate the report.
6. Ensure that the output directory exists and that all referenced files and figures can be accessed.

## Academic Context

**Course:** Introduction to Data Science (IDS)  
**Project:** Credit Card Fraud Detection Using Machine Learning in R  
**Review:** Review 2 — Implementation & Results

## References

1. Machine Learning Group — ULB and Worldline. Credit Card Fraud Detection dataset, available through Kaggle: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. R Project for Statistical Computing: https://www.r-project.org/
3. XGBoost documentation: https://xgboost.readthedocs.io/

---

This project is an academic case study demonstrating machine learning techniques for imbalanced fraud classification. Reported metrics describe the experiments presented in the project and should not be interpreted as a guarantee of real-world fraud detection performance.
