import pandas as pd
import os

def clean_data(data_path):
    print("Loading data...")
    data_path = "C:\\Users\\MOHIT\\OneDrive\\Desktop\\Churn-Prediction\\data\\Telco-Customer-Churn.csv"
    df = pd.read_csv(data_path)
    
    print("\nInitial Dataset Info:")
    print(df.info())
    
    # Handle missing values
    print("\nHandling Missing Values...")
    print("Missing values before handling:")
    print(df.isnull().sum())
    df.ffill(inplace=True)  # Forward-fill missing values as an example
    print("Missing values after handling:")
    print(df.isnull().sum())
    
    # Remove duplicates
    print("\nRemoving Duplicates...")
    initial_shape = df.shape
    df.drop_duplicates(inplace=True)
    final_shape = df.shape
    print(f"Removed {initial_shape[0] - final_shape[0]} duplicate rows.")
    
    # Encoding categorical variables
    print("\nEncoding Categorical Variables...")
    categorical_cols = df.select_dtypes(include=['object']).columns
    print(f"Categorical columns identified: {categorical_cols}")
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    print("\nData after encoding:")
    print(df.head())
    
    # Handle missing values in numeric columns
    print("\nHandling Missing Values in Numeric Columns...")
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        if col in df.columns:
            df.fillna({col: df[col].mean()}, inplace=True)
    
    print("\nFinal Dataset Info:")
    print(df.info())
    
    return df

if __name__ == "__main__":
    # Define the dataset path
    data_path = "../data/Telco-Customer-Churn.csv"
    
    # Clean the data
    cleaned_data = clean_data(data_path)
    
    # Ensure the output directory exists
    output_dir = "C:\\Users\\MOHIT\\OneDrive\\Desktop\\Churn-Prediction\\data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the cleaned data
    cleaned_data.to_csv(f"{output_dir}/cleaned_data.csv", index=False)
    print("\nCleaned data saved to '../data/cleaned_data.csv'")
