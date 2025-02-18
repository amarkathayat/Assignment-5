import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = diabetes.target  # Disease progression

# Check correlation with the target variable
correlations = X.corrwith(pd.Series(y))
correlations_sorted = correlations.abs().sort_values(ascending=False)
print("Feature correlations with target:\n", correlations_sorted)

# Initial model with bmi and s5 (top two correlated features)
X_baseline = X[['bmi', 's5']]
X_train, X_test, y_train, y_test = train_test_split(X_baseline, y, test_size=0.2, random_state=42)

model_baseline = LinearRegression()
model_baseline.fit(X_train, y_train)
y_pred_baseline = model_baseline.predict(X_test)

# Performance of baseline model
mse_baseline = mean_squared_error(y_test, y_pred_baseline)
r2_baseline = r2_score(y_test, y_pred_baseline)
print(f"Baseline Model - MSE: {mse_baseline:.2f}, R²: {r2_baseline:.4f}")

# Add next best feature (excluding bmi and s5)
next_feature = correlations_sorted.index[2]  # Pick third highest correlated feature
X_extended = X[['bmi', 's5', next_feature]]
X_train, X_test, y_train, y_test = train_test_split(X_extended, y, test_size=0.2, random_state=42)

model_extended = LinearRegression()
model_extended.fit(X_train, y_train)
y_pred_extended = model_extended.predict(X_test)

# Performance of extended model
mse_extended = mean_squared_error(y_test, y_pred_extended)
r2_extended = r2_score(y_test, y_pred_extended)
print(f"Extended Model (with {next_feature}) - MSE: {mse_extended:.2f}, R²: {r2_extended:.4f}")

# Further extension (adding more variables)
X_more_features = X[['bmi', 's5', next_feature] + correlations_sorted.index[3:5].tolist()]
X_train, X_test, y_train, y_test = train_test_split(X_more_features, y, test_size=0.2, random_state=42)

model_more = LinearRegression()
model_more.fit(X_train, y_train)
y_pred_more = model_more.predict(X_test)

# Performance with more variables
mse_more = mean_squared_error(y_test, y_pred_more)
r2_more = r2_score(y_test, y_pred_more)
print(f"Model with More Features - MSE: {mse_more:.2f}, R²: {r2_more:.4f}")

# Conclusion
"""
- The feature with the highest correlation (excluding bmi and s5) was added.
- If R² increases and MSE decreases, the new variable improves the model.
- Adding more variables beyond a certain point may not always help.
"""
