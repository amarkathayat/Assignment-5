import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 0: Load dataset
file_path = "50_Startups.csv"  # Update the file path if needed
df = pd.read_csv(file_path)

# Step 1: Identify variables
print("Dataset Preview:\n", df.head())
print("\nColumn Names:", df.columns)

# Step 2: Investigate correlation
df_numeric = df.select_dtypes(include=[np.number])  # Select only numerical columns
correlation_matrix = df_numeric.corr()
print("\nCorrelation Matrix:\n", correlation_matrix)

# Visualize correlation
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Step 3: Choose explanatory variables
"""
I choose R&D Spend, Administration, and Marketing Spend as potential predictors of profit.
R&D Spend has the highest correlation with Profit, followed by Marketing Spend.
Administration has a weaker correlation but might still contribute.
"""
features = ["R&D Spend", "Marketing Spend"]  # Selecting the most correlated variables
target = "Profit"

# Step 4: Plot variables against Profit
plt.figure(figsize=(10, 4))
for feature in features:
    plt.scatter(df[feature], df[target], label=feature)
    plt.xlabel(feature)
    plt.ylabel("Profit")
    plt.title(f"{feature} vs Profit")
    plt.show()

# Step 5: Train-test split (80/20)
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Step 7: Compute RMSE and R²
def evaluate(y_true, y_pred, dataset_type="Train"):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{dataset_type} RMSE: {rmse:.2f}, R²: {r2:.4f}")

evaluate(y_train, y_train_pred, "Train")
evaluate(y_test, y_test_pred, "Test")

# Conclusion
"""
- R&D Spend and Marketing Spend were chosen due to their high correlation with Profit.
- Linear relationships were confirmed through scatter plots.
- Model performance is evaluated using RMSE and R².
- If Test R² is significantly lower than Train R², overfitting might be present.
"""
