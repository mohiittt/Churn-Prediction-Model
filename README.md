# Churn Prediction Model

This repository contains a machine learning project for predicting customer churn using data from a telecommunications company.

## Project Overview

Customer churn is a significant challenge for businesses, particularly in competitive industries like telecommunications. This project aims to develop a predictive model to identify customers likely to churn and provide actionable insights to reduce churn rates.

## Dataset

The project uses the **Telco Customer Churn** dataset, which includes customer demographics, account information, and service usage data. The dataset contains the following key features:

- **CustomerID**: Unique identifier for each customer.
- **Demographic Details**: Gender, Senior Citizen status, Partner, and Dependents.
- **Account Information**: Tenure, Monthly Charges, Total Charges.
- **Services**: Internet Service, Streaming, Device Protection, etc.
- **Target Variable**: Churn (Yes/No).

### Files

- `data/Telco-Customer-Churn.csv`: Raw dataset.
- `data/cleaned_data.csv`: Preprocessed dataset after cleaning.
- `scripts/preprocessing.py`: Data cleaning and preprocessing script.
- `scripts/modeling.py`: Code for training and evaluating models.
- `scripts/load_data.py`: Helper functions for data loading.
- `models/random_forest_model.pkl`: Trained Random Forest model.
- `feature_importance.png`: Visual representation of feature importance.

## Key Steps

1. **Data Cleaning and Preprocessing**:

   - Handling missing values.
   - Encoding categorical variables.
   - Scaling numerical features.

2. **Exploratory Data Analysis (EDA)**:

   - Visualizing churn distribution.
   - Analyzing feature correlations.

3. **Model Building and Evaluation**:

   - Training a Random Forest classifier.
   - Evaluating performance using metrics like Accuracy, Precision, Recall, and F1-Score.

4. **Feature Importance**:
   - Identifying key features influencing churn using feature importance from the model.

## Results

The Random Forest model achieved the following performance on the test dataset:

- **Accuracy**: 85%
- **Precision**: 80%
- **Recall**: 75%
- **F1-Score**: 77%

## Usage

To run the project locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/mohiittt/Churn-Prediction-Model.git
   cd Churn-Prediction-Model
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run preprocessing:
   ```bash
   python scripts/preprocessing.py
   ```
4. Train the model:
   ```bash
   python scripts/modeling.py
   ```

## Future Work

- Experiment with other machine learning algorithms (e.g., Gradient Boosting, Neural Networks).
- Implement hyperparameter tuning for better performance.
- Develop a web application for deployment.

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests.

## License

This project is licensed under the MIT License.

---

For any questions, please contact [mohiittt](https://github.com/mohiittt).
