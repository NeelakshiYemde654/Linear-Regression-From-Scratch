
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from linear_regression import gradient_descent, predict


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("dataset/housing.csv")

features = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income"
]

X = df[features]
y = df["median_house_value"]


# ==========================================
# 2. SPLIT DATASET
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

y_train_original = y_train.to_numpy()
y_test = y_test.to_numpy()


# ==========================================
# 3. HANDLE MISSING VALUES
# ==========================================

imputer = SimpleImputer(strategy="median")

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)


# ==========================================
# 4. STANDARDIZE FEATURES
# ==========================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ==========================================
# 5. STANDARDIZE TARGET
# ==========================================

y_scaler = StandardScaler()

y_train = y_scaler.fit_transform(
    y_train_original.reshape(-1, 1)
).flatten()


# ==========================================
# 6. EVALUATION FUNCTIONS
# ==========================================

def calculate_mse(y_actual, y_predicted):
    return np.mean((y_actual - y_predicted) ** 2)


def calculate_r2(y_actual, y_predicted):

    ss_res = np.sum((y_actual - y_predicted) ** 2)

    ss_total = np.sum(
        (y_actual - np.mean(y_actual)) ** 2
    )

    return 1 - (ss_res / ss_total)


def evaluate_model(name, y_actual, y_predicted):

    mse = calculate_mse(y_actual, y_predicted)
    r2 = calculate_r2(y_actual, y_predicted)

    print(f"\n===== {name} EVALUATION =====")
    print("MSE:", mse)
    print("R2 Score:", r2)

    return mse, r2


# ==========================================
# 7. SIMPLE LINEAR REGRESSION
# ==========================================

print("\n========== SIMPLE LINEAR REGRESSION ==========")

# Use only median_income as the input feature.
income_index = features.index("median_income")

X_train_simple = X_train[:, [income_index]]
X_test_simple = X_test[:, [income_index]]

# Add bias column.
X_train_simple = np.c_[
    np.ones(X_train_simple.shape[0]),
    X_train_simple
]

X_test_simple = np.c_[
    np.ones(X_test_simple.shape[0]),
    X_test_simple
]

# Initialize parameters.
theta_simple = np.zeros(X_train_simple.shape[1])

# Train using gradient descent.
theta_simple, simple_cost_history = gradient_descent(
    X_train_simple,
    y_train,
    theta_simple,
    learning_rate=0.03,
    iterations=2000
)

print("\nSimple Model Parameters:")
print(theta_simple)

print("Initial Cost:", simple_cost_history[0])
print("Final Cost:", simple_cost_history[-1])

# Generate predictions.
simple_predictions_scaled = predict(
    X_test_simple,
    theta_simple
)

simple_predictions = y_scaler.inverse_transform(
    simple_predictions_scaled.reshape(-1, 1)
).flatten()

# Evaluate NumPy model.
simple_mse, simple_r2 = evaluate_model(
    "SIMPLE NUMPY MODEL",
    y_test,
    simple_predictions
)

# Compare with Scikit-learn.
simple_sk_model = LinearRegression()

simple_sk_model.fit(
    X_train_simple[:, 1:],
    y_train_original
)

simple_sk_predictions = simple_sk_model.predict(
    X_test_simple[:, 1:]
)

evaluate_model(
    "SIMPLE SCIKIT-LEARN MODEL",
    y_test,
    simple_sk_predictions
)


# ==========================================
# 8. MULTIPLE LINEAR REGRESSION
# ==========================================

print("\n========== MULTIPLE LINEAR REGRESSION ==========")

# Add bias column.
X_train_multiple = np.c_[
    np.ones(X_train.shape[0]),
    X_train
]

X_test_multiple = np.c_[
    np.ones(X_test.shape[0]),
    X_test
]

# Initialize parameters.
theta_multiple = np.zeros(X_train_multiple.shape[1])

# Train using gradient descent.
theta_multiple, multiple_cost_history = gradient_descent(
    X_train_multiple,
    y_train,
    theta_multiple,
    learning_rate=0.03,
    iterations=2000
)

print("\nMultiple Model Parameters:")
print(theta_multiple)

print("Initial Cost:", multiple_cost_history[0])
print("Final Cost:", multiple_cost_history[-1])

# Generate predictions.
multiple_predictions_scaled = predict(
    X_test_multiple,
    theta_multiple
)

multiple_predictions = y_scaler.inverse_transform(
    multiple_predictions_scaled.reshape(-1, 1)
).flatten()

# Evaluate NumPy model.
multiple_mse, multiple_r2 = evaluate_model(
    "MULTIPLE NUMPY MODEL",
    y_test,
    multiple_predictions
)

# Compare with Scikit-learn.
multiple_sk_model = LinearRegression()

multiple_sk_model.fit(
    X_train,
    y_train_original
)

multiple_sk_predictions = multiple_sk_model.predict(X_test)

multiple_sk_mse, multiple_sk_r2 = evaluate_model(
    "MULTIPLE SCIKIT-LEARN MODEL",
    y_test,
    multiple_sk_predictions
)


# ==========================================
# 9. MODEL COMPARISON
# ==========================================

print("\n========== MODEL COMPARISON ==========")

print("\nSimple Linear Regression:")
print(f"NumPy MSE: {simple_mse:.2f}")
print(f"NumPy R2:  {simple_r2:.6f}")

print("\nMultiple Linear Regression:")
print(f"NumPy MSE: {multiple_mse:.2f}")
print(f"NumPy R2:  {multiple_r2:.6f}")

print("\nScikit-learn Multiple Regression:")
print(f"MSE: {multiple_sk_mse:.2f}")
print(f"R2:  {multiple_sk_r2:.6f}")


# ==========================================
# 10. COEFFICIENT COMPARISON
# ==========================================

our_coefficients = (
    theta_multiple[1:] * y_scaler.scale_[0]
)

our_intercept = (
    y_scaler.mean_[0]
    + theta_multiple[0] * y_scaler.scale_[0]
)

print("\n========== MULTIPLE MODEL COEFFICIENTS ==========")

print("\nNumPy Coefficients:")
print(our_coefficients)

print("\nScikit-learn Coefficients:")
print(multiple_sk_model.coef_)

print("\nNumPy Intercept:", our_intercept)
print("Scikit-learn Intercept:", multiple_sk_model.intercept_)


# ==========================================
# 11. COST CONVERGENCE GRAPHS
# ==========================================

os.makedirs("results", exist_ok=True)

# Simple regression graph.
plt.figure(figsize=(8, 5))

plt.plot(simple_cost_history)

plt.xlabel("Iterations")
plt.ylabel("Mean Squared Error")
plt.title("Simple Linear Regression - Cost Convergence")

plt.grid(True)

plt.savefig("results/simple_cost_convergence.png")
plt.close()


# Multiple regression graph.
plt.figure(figsize=(8, 5))

plt.plot(multiple_cost_history)

plt.xlabel("Iterations")
plt.ylabel("Mean Squared Error")
plt.title("Multiple Linear Regression - Cost Convergence")

plt.grid(True)

plt.savefig("results/cost_convergence.png")
plt.close()


print("\nBoth cost convergence graphs saved in the results folder.")

print("\nProject execution completed successfully!")
