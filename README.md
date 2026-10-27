# Employee Attrition Prediction and Data Drift Monitoring using MLflow

## 1. Project Overview

This project demonstrates an MLOps workflow for employee attrition prediction and data drift monitoring. A Random Forest classification model is trained using employee data, and MLflow is used to track model parameters, performance metrics, and model artifacts.

A statistical test is used to compare reference employee data with simulated current data to identify changes in feature distributions.

## 2. Objectives

- Train a machine learning model to predict employee attrition.
- Track experiments using MLflow.
- Evaluate the model using accuracy, precision, recall, and F1-score.
- Detect data drift using the Kolmogorov-Smirnov statistical test.
- Record drift analysis results in MLflow.

## 3. Technologies Used

- Python 3.11.9
- MLflow 3.17.2
- Pandas
- Scikit-learn
- SciPy
- NumPy
- Matplotlib

## 4. Dataset

The project uses the IBM HR Analytics Employee Attrition dataset.

- Total records: 1,470
- Total columns: 35
- Missing values: 0
- Target variable: Attrition

The target indicates whether an employee left the organization.

## 5. Project Files

- `employee_attrition.csv` - Employee dataset
- `check_dataset.py` - Dataset inspection script
- `train.py` - Model training and MLflow experiment tracking
- `drift_analysis.py` - Data drift detection
- `drift_report.csv` - Drift analysis results
- `requirements.txt` - Python dependencies
- `mlflow.db` - MLflow tracking database

## 6. Installation and Execution

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Inspect the dataset:

```powershell
python check_dataset.py
```

Train the model and track the experiment:

```powershell
python train.py
```

Run drift analysis:

```powershell
python drift_analysis.py
```

Launch the MLflow dashboard:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open http://127.0.0.1:5000 in a browser.

## 7. Model Evaluation Results

The Random Forest model achieved the following results on the test dataset:

- Accuracy: 0.8333
- Precision: 0.4688
- Recall: 0.3191
- F1-score: 0.3797

The model identifies employees who stayed more effectively than employees who left. Further model improvement may be required.

## 8. Data Drift Analysis

The Kolmogorov-Smirnov (KS) test was used with a significance threshold of 0.05.

The reference dataset contained 1,470 records. The simulated current dataset also contained 1,470 records.

Drift was detected in:

- Age
- MonthlyIncome

No statistically significant drift was detected in:

- DistanceFromHome
- YearsAtCompany
- TotalWorkingYears

The drift was simulated by deliberately changing selected feature distributions. These results demonstrate the detection method and do not represent naturally observed employee data drift.

## 9. Conclusion

The project demonstrates model training, experiment tracking, artifact logging, and statistical data drift detection using MLflow and Python.

The results show that controlled changes in employee feature distributions can be detected using the KS test. In a real deployment, detected drift should trigger further investigation and model performance evaluation before deciding whether retraining is necessary.
