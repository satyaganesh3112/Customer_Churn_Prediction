# 📊 Telco Customer Churn Prediction

Predicting customer behavior is critical for telecommunications companies to reduce turnover and increase revenue. This project uses machine learning to identify customers at high risk of churning, allowing for targeted retention strategies.



## 📋 Project Overview
This project involves building a classification model to determine whether a customer will churn based on various demographic, account, and service-related features. 

The primary model used in this notebook is the **Random Forest Classifier**, optimized through hyperparameter tuning to ensure the highest possible **Recall** for identifying potential churners.



## 📊 Dataset
The dataset used is the **WA_Fn-UseC_-Telco-Customer-Churn.csv**, which contains **7,043 rows** and **21 columns**.

### Key Features:
* **Demographics**: Gender, Senior Citizen status, Partner, and Dependents.
* **Services**: Phone service, Multiple lines, Internet service (DSL/Fiber optic), Online security, Tech support, Streaming TV/Movies.
* **Account Info**: Tenure, Contract type, Payment method, Paperless billing, Monthly charges, and Total charges.
* **Target**: `Churn` (Yes/No).

## 🛠️ Tech Stack
* **Language**: Python 3.x
* **Libraries**:
    * **Data Manipulation**: `pandas`, `numpy`
    * **Visualization**: `matplotlib`, `seaborn`
    * **Machine Learning**: `scikit-learn`

## 🚀 Project Workflow

### 1. Data Cleaning & Preprocessing
* **Dropping Unnecessary Columns**: Removed `customerID` as it doesn't contribute to the model's predictive power.
* **Type Conversion**: Converted `TotalCharges` to a numeric format (handling empty strings as NaN).
* **Missing Value Handling**: Managed missing values resulting from zero-tenure customers.

### 2. Exploratory Data Analysis (EDA)
* Visualized the distribution of churn across different service types and demographics.
* Analyzed the correlation between monthly charges, tenure, and churn rates.

### 3. Model Building
* **Algorithm**: `RandomForestClassifier`.
* **Data Splitting**: Used `train_test_split` (Train/Test) to create training and testing sets.
* **Cross-Validation**: Implemented `StratifiedKFold` and `cross_val_score` to ensure model robustness.

### 4. Hyperparameter Tuning
* Utilized `RandomizedSearchCV` to find the optimal parameters for the Random Forest model, specifically tuning for better performance on the minority class (churners).

### 5. Evaluation
The model is evaluated using a combination of metrics to ensure it captures at-risk customers effectively.



## 📈 Final Results

The model was optimized to prioritize **Recall** for the churn class (Class 1) to ensure the business doesn't "miss" customers who are about to leave.

### Performance Metrics:
| Metric | Class 0 (Stayed) | Class 1 (Churned) | Overall |
| :--- | :--- | :--- | :--- |
| **Precision** | 0.92 | 0.56 | **Accuracy: 78%** |
| **Recall** | 0.77 | **0.82** | **Macro Avg: 79%** |
| **F1-Score** | 0.84 | 0.67 | |

### Key Insights:
* **Recall of 82%**: The model successfully identifies 82% of all actual churners.
* **Feature Importance**: **Tenure**, **Contract Type**, and **Monthly Charges** were found to be the most influential factors in predicting churn.


