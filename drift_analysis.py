
import pandas as pd
import numpy as np
import mlflow

from scipy.stats import ks_2samp

# 1. Load the employee dataset
df = pd.read_csv("employee_attrition.csv")

# Select numerical features for drift analysis
features = [
    "Age",
    "MonthlyIncome",
    "DistanceFromHome",
    "YearsAtCompany",
    "TotalWorkingYears"
]

# 2. Create reference data
# Use the original employee records as reference data
reference_data = df[features].copy()

# 3. Simulate current data with changes in selected features
# Fixed random seed ensures reproducible results
rng = np.random.default_rng(42)

current_data = reference_data.copy()

# Simulate an increase in monthly income
current_data["MonthlyIncome"] = (
    current_data["MonthlyIncome"] * 1.30
).round()

# Simulate a change in employee age distribution
current_data["Age"] = (
    current_data["Age"] + rng.integers(1, 6, size=len(current_data))
).clip(18, 65)

# 4. Detect drift using the Kolmogorov-Smirnov test
results = []

for feature in features:
    statistic, p_value = ks_2samp(
        reference_data[feature],
        current_data[feature]
    )

    drift_detected = p_value < 0.05

    results.append({
        "feature": feature,
        "ks_statistic": statistic,
        "p_value": p_value,
        "drift_detected": drift_detected
    })

results_df = pd.DataFrame(results)

# 5. Display results
print("\nDATA DRIFT ANALYSIS RESULTS")
print(results_df.to_string(index=False))

print("\nReference records:", len(reference_data))
print("Current records:", len(current_data))
print("Significance threshold: 0.05")
print("Drifted features:", int(results_df["drift_detected"].sum()))

# 6. Save the report
results_df.to_csv("drift_report.csv", index=False)

# 7. Log drift analysis in MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Employee_Attrition_Drift_Analysis")

with mlflow.start_run(run_name="Simulated_Data_Drift"):
    mlflow.log_param("drift_method", "Kolmogorov-Smirnov test")
    mlflow.log_param("significance_threshold", 0.05)
    mlflow.log_param("reference_records", len(reference_data))
    mlflow.log_param("current_records", len(current_data))
    mlflow.log_param("data_scenario", "Simulated feature distribution changes")

    mlflow.log_metric(
        "drifted_feature_count",
        int(results_df["drift_detected"].sum())
    )

    for _, row in results_df.iterrows():
        mlflow.log_metric(
            f"{row['feature']}_ks_statistic",
            float(row["ks_statistic"])
        )
        mlflow.log_metric(
            f"{row['feature']}_p_value",
            float(row["p_value"])
        )

    mlflow.log_artifact("drift_report.csv")

print("\nDrift report saved to drift_report.csv")
print("Drift results logged successfully in MLflow.")
