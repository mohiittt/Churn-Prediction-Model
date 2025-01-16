import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.feature_selection import VarianceThreshold

# Load cleaned data
print("Loading cleaned data...")
data_path = "C:\\Users\\MOHIT\\OneDrive\\Desktop\\Churn-Prediction\\data\\cleaned_data.csv"
df = pd.read_csv(data_path)

print("\nInitial Data Info:")
print(df.info())

# Separate Features and Target
target = "Churn_Yes"
if target not in df.columns:
    raise KeyError(f"Target column '{target}' not found in dataset.")
X = df.drop(columns=[target], errors="ignore")
y = df[target]

# Drop Low-Variance Features
print("\nDropping low-variance features...")
selector_var = VarianceThreshold(threshold=0.01)  # Adjust threshold as needed
X_var_filtered = selector_var.fit_transform(X)
retained_columns = X.columns[selector_var.get_support()]
X = pd.DataFrame(X_var_filtered, columns=retained_columns)

print(f"Reduced dataset shape: {X.shape}")

# Compute Correlation Matrix for Reduced Data
print("\nComputing correlation matrix...")
correlation_matrix = X.corr()

# Identify Highly Correlated Features
print("\nIdentifying highly correlated features...")
correlated_features = set()
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.9:  # Adjust threshold if needed
            colname = correlation_matrix.columns[i]
            correlated_features.add(colname)

print(f"Highly correlated features to drop: {len(correlated_features)}")
print(correlated_features)

# Drop Highly Correlated Features
X.drop(columns=correlated_features, inplace=True, errors='ignore')

# Perform Feature Selection
print("\nPerforming feature selection...")
selector = SelectKBest(score_func=mutual_info_classif, k=20)  # Adjust k as needed
X_selected = selector.fit_transform(X, y)

selected_features = X.columns[selector.get_support()]
print(f"Selected features: {list(selected_features)}")

# Train-Test Split
print("\nSplitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)

# Train a Random Forest Classifier
print("\nTraining Random Forest Classifier...")
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate the Model
print("\nEvaluating model...")
y_pred = rf_model.predict(X_test)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2f}")

# Save the Model
import joblib
model_path = "C:\\Users\\MOHIT\\OneDrive\\Desktop\\Churn-Prediction\\models\\random_forest_model.pkl"
print(f"\nSaving model to {model_path}...")
joblib.dump(rf_model, model_path)

print("\nModeling completed successfully!")

# Feature Importance Analysis (This is optional and requires additional setup)

from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np

# Train the Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=42, n_estimators=100)
rf_classifier.fit(X_train, y_train)

# Feature Importance Analysis
print("Analyzing feature importance...")
feature_importances = rf_classifier.feature_importances_
sorted_indices = np.argsort(feature_importances)[::-1]

# Top 10 features
top_features = 10
top_indices = sorted_indices[:top_features]
top_importances = feature_importances[top_indices]

# Plotting feature importances
plt.figure(figsize=(10, 6))
plt.barh(range(top_features), top_importances[::-1], align='center')
plt.yticks(range(top_features), [selected_features[i] for i in top_indices[::-1]])
plt.xlabel('Feature Importance')
plt.title('Top 10 Feature Importances')
plt.tight_layout()
plt.savefig("feature_importance.png")  # Save for review
plt.show()

# Hyperparameter Tuning (Again Optional)

from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Grid Search
print("Starting hyperparameter tuning...")
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, 
                           scoring='accuracy', cv=5, verbose=1, n_jobs=-1)

grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_

print(f"Best Parameters: {best_params}")

# Evaluate the best model
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)

# Compute accuracy and classification report for the tuned model
print("\nEvaluation of the best-tuned model:")
print("Accuracy:", accuracy_score(y_test, y_pred_best))
print("Classification Report:\n", classification_report(y_test, y_pred_best))
