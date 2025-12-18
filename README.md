# Customer_Churn_Prediction
Telco Customer Churn Prediction
This project aims to predict customer churn for a telecommunications company using machine learning. By identifying customers at high risk of leaving, the company can take proactive measures to improve retention.

📋 Project Overview
The project involves building a classification model to determine whether a customer will churn based on various demographic, account, and service-related features.

The primary model used in this notebook is the Random Forest Classifier, optimized through hyperparameter tuning.

📊 Dataset
The dataset used is the WA_Fn-UseC_-Telco-Customer-Churn.csv, which contains 7,043 rows and 21 columns.

Key Features:
Demographics: Gender, Senior Citizen status, Partner, and Dependents.

Services: Phone service, Multiple lines, Internet service (DSL/Fiber optic), Online security, Tech support, Streaming TV/Movies.

Account Info: Tenure, Contract type, Payment method, Paperless billing, Monthly charges, and Total charges.

Target: Churn (Yes/No).

🛠️ Tech Stack
Language: Python

Libraries:

Data Manipulation: pandas, numpy

Visualization: matplotlib, seaborn

Machine Learning: scikit-learn

🚀 Project Workflow
1. Data Cleaning & Preprocessing
Dropping Unnecessary Columns: Removed customerID as it doesn't contribute to the model's predictive power.

Type Conversion: Converted TotalCharges to a numeric format (handling empty strings as NaN).

Missing Value Handling: Identified and managed missing values resulting from zero-tenure customers.

2. Exploratory Data Analysis (EDA)
Visualizing the distribution of churn across different service types and demographics.

Analyzing the correlation between monthly charges, tenure, and churn rates.

3. Model Building
Algorithm: RandomForestClassifier.

Data Splitting: Used train_test_split to create training and testing sets.

Cross-Validation: Implemented StratifiedKFold and cross_val_score to ensure model robustness.

4. Hyperparameter Tuning
Utilized RandomizedSearchCV to find the optimal parameters for the Random Forest model.

5. Evaluation
The model is evaluated using several metrics:

Accuracy Score

Confusion Matrix

Classification Report (Precision, Recall, F1-Score)

📈 Final ResultsThe model was optimized to prioritize Recall for the churn class (Class 1) to ensure the business doesn't "miss" customers who are about to leave.Performance Metrics:MetricClass 0 (Stayed)Class 1 (Churned)OverallPrecision0.920.56Accuracy: 78%Recall0.770.82Macro Avg: 79%F1-Score0.840.67Key Insights:Recall of 82%: The model successfully identifies 82% of all actual churners.Feature Importance: Tenure, Contract Type, and Monthly Charges were the most influential factors in predicting churn.
