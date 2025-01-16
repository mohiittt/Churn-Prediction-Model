import pandas as pd

# Load the dataset
data_path = "C:\\Users\\MOHIT\\OneDrive\\Desktop\\Churn-Prediction\\data\\Telco-Customer-Churn.csv"
df = pd.read_csv(data_path)

print(df.head())

# Dataset information
print("\nDataset Information:")
print(df.info())

# Missing values
print("\nMissing Values in Each Column:")
print(df.isnull().sum())

# Statistical summary
print("\nStatistical Summary:")
print(df.describe())

#  Duplicate rows
print("\nNumber of Duplicate Rows:", df.duplicated().sum())
