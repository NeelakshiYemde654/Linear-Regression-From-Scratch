
import numpy as np


# 1. Hypothesis Function
def predict(X, theta):
    return np.dot(X, theta)


# 2. Mean Squared Error Cost Function
def compute_cost(X, y, theta):
    m = len(y)

    predictions = predict(X, theta)

    errors = predictions - y

    cost = np.sum(errors ** 2) / m

    return cost


# 3. Gradient Calculation
def compute_gradient(X, y, theta):
    m = len(y)

    predictions = predict(X, theta)

    errors = predictions - y

    gradient = (2 / m) * np.dot(X.T, errors)

    return gradient


# 4. Gradient Descent
def gradient_descent(X, y, theta, learning_rate, iterations):

    cost_history = []

    for i in range(iterations):

        gradient = compute_gradient(X, y, theta)

        theta = theta - learning_rate * gradient

        cost = compute_cost(X, y, theta)

        cost_history.append(cost)

    return theta, cost_history
