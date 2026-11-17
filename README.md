# Bank Marketing Subscription Prediction

An end-to-end Machine Learning project that compares seven classification algorithms to predict whether a customer will subscribe to a bank term deposit using customer and campaign-related information.

## Overview

Bank marketing campaigns involve contacting potential customers to promote term deposits. However, contacting every customer can be costly and inefficient.

This project uses supervised machine learning to identify customers who are more likely to subscribe, helping banks make more focused marketing decisions.

The project follows a complete ML workflow, from data exploration and preprocessing to model comparison, evaluation, and reusable model export.

## Problem Statement

Predict whether a bank customer will subscribe to a term deposit (`yes` or `no`) using customer attributes and historical campaign information available before a marketing call.

**Dataset:** UCI Bank Marketing Dataset  
**Records:** 45,211 (bank-full dataset)  
**Target Variable:** `deposit`  
**Problem Type:** Binary Classification

## Machine Learning Algorithms

Seven classification algorithms are implemented and compared:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting
5. K-Nearest Neighbours (KNN)
6. Gaussian Naive Bayes
7. Linear Support Vector Machine (SVM)

## Project Workflow

- **Data Collection:** Retrieved the Bank Marketing dataset from the UCI Machine Learning Repository.
- **Data Understanding:** Examined data types, duplicate records, target distribution, and data quality.
- **Exploratory Data Analysis:** Used statistical analysis and visualizations to explore customer characteristics and subscription patterns.
- **Data Preprocessing:** Handled missing categorical values, encoded categorical variables, imputed numerical values, and standardized numerical features.
- **Feature Engineering:** Created a `previously_contacted` feature and handled the special `pdays = -1` value.
- **Model Training:** Trained seven classification algorithms using Scikit-learn pipelines.
- **Model Evaluation:** Applied stratified 5-fold cross-validation and a separate 80:20 train-test split.
- **Threshold Optimization:** Compared models using individually optimized decision thresholds.
- **Model Interpretation:** Used permutation importance to understand feature contributions.
- **Model Export:** Saved the trained pipeline, prediction threshold, and metadata using Joblib.

## Evaluation Metrics

Since the dataset has an imbalanced target distribution, accuracy alone is not sufficient.

The following metrics are used:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- PR-AUC
- Confusion Matrix
- Brier Score (for probability calibration)

## Visualizations

The project includes:

- Customer subscription distribution
- Customer demographic analysis
- Account balance distributions
- Feature correlation analysis
- Interactive 2D visualizations using Plotly
- Interactive 3D and 4D visualizations
- Classification model comparison dashboard
- Radar chart for model comparison
- ROC and Precision-Recall curves
- Confusion matrix
- Feature importance visualization

## Tech Stack

| Category | Technologies |
|---|---|
| Language | Python |
| Data Analysis | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Visualization | Matplotlib, Seaborn, Plotly |
| Statistical Analysis | SciPy |
| Model Serialization | Joblib |
| Dataset | UCI Machine Learning Repository |
| Environment | Google Colab / Jupyter Notebook |

## Project Structure

```text
bank-marketing-ml-prediction/
│
├── Group1_AlgorithmShowdown_Notebook_Revised.ipynb
├── requirements.txt
├── bank_term_deposit_model.joblib
├── model_metadata.json
└── README.md
```

The model artifacts and requirements file are generated when the notebook is executed.

## Installation and Usage

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/bank-marketing-ml-prediction.git
```

### 2. Navigate to the project directory

```bash
cd bank-marketing-ml-prediction
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the notebook

Open the notebook using Jupyter Notebook or Google Colab and execute the cells sequentially.

The dataset is retrieved directly from the UCI repository using the `ucimlrepo` package.

### 5. Generate predictions

After executing the model export and prediction sections, the trained model can be loaded using Joblib and used to predict subscriptions for new customer records.

## Model Export

The project exports the following files:

- `bank_term_deposit_model.joblib` – Contains the trained ML pipeline, selected decision threshold, and feature information.
- `model_metadata.json` – Stores model configuration and metadata.
- `requirements.txt` – Contains the installed package versions for reproducibility.

## Important Considerations

- The prediction is designed for use before a marketing call.
- `duration` and `campaign` are excluded to reduce information leakage.
- Class imbalance is considered during model evaluation and threshold optimization.
- The dataset represents a particular Portuguese bank and campaign period, so results may not generalize to other banking environments.
- Predictive feature importance indicates association, not causation.
- Model probabilities may require calibration before being interpreted as actual subscription probabilities.

## Dataset Source

UCI Machine Learning Repository:

https://archive.ics.uci.edu/dataset/222/bank+marketing

## Author

**Abhishek**

B.Sc. Artificial Intelligence and Machine Learning  
University of Mumbai

---

*Built as part of an academic Machine Learning project exploring classification algorithms, model evaluation, and practical predictive analytics.*
