# Machine Learning System for Fitness Level Classification

A machine learning project focused on predicting a user's fitness level (Low, Moderate, or High) from longitudinal activity, sleep, and health data.

## Overview

We built this project to explore how well machine learning models can identify fitness levels from a combination of behavioral and physiological indicators. Rather than relying on a single metric, the system uses activity patterns, sleep quality, health measurements, and engineered features to make predictions.

The project compares several machine learning models under the same preprocessing and validation pipeline, with a focus on preventing data leakage in user-based datasets. Since each individual contributes multiple observations, special care was taken to ensure that information from the same user never appeared across training and evaluation partitions.

## Dataset

The dataset is a synthetic longitudinal health dataset containing 90,000 records from 3,000 individuals, with 30 daily observations per user.

### Included Features

- Activity metrics
  - Steps
  - Move Minutes
  - Heart Points

- Sleep metrics
  - Sleep Hours
  - Sleep Efficiency

- Health measurements
  - BMI
  - Blood Pressure
  - Blood Glucose

### Target Variable

| Class | Fitness Level |
|---------|---------|
| 0 | Low |
| 1 | Moderate |
| 2 | High |

## Methodology

### Data Preprocessing

Before training, the data was processed through several stages:

- User-based train/test splitting using GroupShuffleSplit
- Median imputation for missing values
- One-hot encoding of activity categories
- Feature scaling for Logistic Regression using StandardScaler

A grouped train/test split was used to ensure that records from the same individual never appeared in both training and testing data.

### Feature Engineering

To capture relationships that are not directly visible in the raw data, several additional features were created:

- `hp_efficiency`
- `pulse_pressure`
- `aerobic_load`
- `glucose_ratio`
- `bmi_cat`

All engineered features were generated using reference statistics computed exclusively from the training partition, helping prevent information leakage from the evaluation data.

## Models Evaluated

The following models were trained and compared:

 - Logistic Regression
 - Random Forest
 - XGBoost
 - LightGBM
 - CatBoost

## Validation Strategy

One of the main goals of this project was to evaluate models realistically.

Because each user appears multiple times in the dataset, traditional random cross-validation can lead to overly optimistic performance estimates. To avoid this, I used GroupKFold Cross-Validation, ensuring that all observations belonging to a particular user remained within the same fold.

This approach produces evaluation results that are much closer to how the model would perform on completely unseen users.

## Results

| Model | Accuracy | F1 Macro | F1 Weighted | ROC-AUC |
|---------|---------:|---------:|---------:|---------:|
| Logistic Regression | 0.6965 | 0.6972 | 0.6925 | 0.8611 |
| Random Forest | 0.7027 | 0.6991 | 0.6949 | 0.8725 |
| XGBoost | **0.7318** | **0.7283** | **0.7301** | 0.8852 |
| LightGBM | 0.7229 | 0.7216 | 0.7200 | 0.8788 |
| CatBoost | 0.7236 | 0.7225 | 0.7188 | **0.8867** |

### Best Performing Model

**XGBoost**

 - Accuracy: **73.18%**
 - F1 Macro: **72.83%**
 - ROC-AUC: **88.52%**

## Visualizations

The project includes several visual analyses:

 - Class distribution and user distribution plots
 - Confusion matrices
 - One-vs-Rest ROC curves
 - Feature importance analysis
 - Model comparison charts
 - Cross-validation results for Logistic Regression hyperparameter tuning

![Confusion Matrix](confusion_matrices.png)

![ROC Curves](roc_curves.png)

![Feature Importance](feature_importance.png)

## Key Findings
 - Gradient boosting models consistently outperformed the linear baseline.
 - Engineered physiological features improved predictive performance beyond the original variables.
 - User-aware validation helped reduce the risk of inflated performance estimates caused by data leakage.
 - Activity intensity and health-related indicators emerged as some of the most influential predictors of fitness level.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- LightGBM
- CatBoost
- Matplotlib
- Seaborn

## Authors

- Riya Gupta
- Shalini Dubey

