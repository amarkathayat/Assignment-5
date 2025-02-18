import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# Step 1: Read the dataset
file_path = "Auto.csv"  # Update the file path if necessary
df = pd.read_csv(file_path)

# Step 2: Prepare X (features) and y (target)
"""
I predict 'mpg' using all numerical variables except 'mpg' itself.
I exclude 'name' and 'origin' since 'name' is categorical and 'origin' might not be numeric.
"""
df = df.dropna()  # Drop missing values if any
X = df.drop(columns=["mpg", "name", "origin"], errors="ignore")
y = df["mpg"]

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 4 & 5: Train Ridge and LASSO models with different alpha values
alphas = np.logspace(-3, 3, 50)  # Try alpha values from 0.001 to 1000
ridge_scores = []
lasso_scores = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    lasso = Lasso(alpha=alpha)

    ridge.fit(X_train, y_train)
    lasso.fit(X_train, y_train)

    ridge_scores.append(r2_score(y_test, ridge.predict(X_test)))
    lasso_scores.append(r2_score(y_test, lasso.predict(X_test)))

# Step 6: Plot R² scores as functions of alpha
plt.figure(figsize=(8, 5))
plt.plot(alphas, ridge_scores, label="Ridge R²", marker="o")
plt.plot(alphas, lasso_scores, label="LASSO R²", marker="s")
plt.xscale("log")  # Use log scale for alpha
plt.xlabel("Alpha (log scale)")
plt.ylabel("R² Score")
plt.title("R² Score vs Alpha for Ridge and LASSO")
plt.legend()
plt.grid(True)
plt.show()

# Step 7: Identify best alpha
best_alpha_ridge = alphas[np.argmax(ridge_scores)]
best_alpha_lasso = alphas[np.argmax(lasso_scores)]
best_r2_ridge = max(ridge_scores)
best_r2_lasso = max(lasso_scores)

print(f"Best Ridge Alpha: {best_alpha_ridge:.4f}, Best R²: {best_r2_ridge:.4f}")
print(f"Best LASSO Alpha: {best_alpha_lasso:.4f}, Best R²: {best_r2_lasso:.4f}")

# Conclusion
"""
- Ridge and LASSO regression were tested with different alpha values.
- The best alpha was selected based on the highest R² score on the test data.
- Ridge tends to handle multicollinearity better, while LASSO can shrink coefficients to zero, removing less useful variables.
- The optimal alpha can be found using the plotted R² scores.
"""
