
import pandas as pd

# Load the employee attrition dataset
df = pd.read_csv("employee_attrition.csv")

# Display basic dataset information
print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

print("\nFirst 5 records:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values in each column:")
print(df.isnull().sum())

print("\nAttrition distribution:")
print(df["Attrition"].value_counts())
