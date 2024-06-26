import numpy as np
import cvxpy as cp
import pandas as pd
import logging

import mlopt
from mlopt.sampling import uniform_sphere_sample
from mlopt.learners import XGBoost
from mlopt.utils import n_features, pandas2array

np.random.seed(1)  # Reset random seed for reproducibility

# Variable
n = 10
x = cp.Variable(n, integer=True)

# Cost
c = np.random.rand(n)

# Weights
a = cp.Parameter(n, nonneg=True, name='a')
x_u = cp.Parameter(n, nonneg=True, name='x_u')
b = 0.5 * n

# Problem
cost = - c @ x
constraints = [a @ x <= b,
               0 <= x, x <= x_u]


# Define optimizer
# If you just want to remove too many messages
# change INFO to WARNING
problem = cp.Problem(cp.Minimize(cost), constraints)
m = mlopt.Optimizer(problem,
                    log_level=logging.INFO)


# Average request
theta_bar = 2 * np.ones(2 * n)
radius = 1.0


def sample(theta_bar, radius, n=100):

    # Sample points from multivariate ball
    ndim = int(len(theta_bar)/2)
    X_a = uniform_sphere_sample(theta_bar[:ndim], radius, n=n)
    X_u = uniform_sphere_sample(theta_bar[ndim:], radius, n=n)

    df = pd.DataFrame({
        'a': list(X_a),
        'x_u': list(X_u)
        })

    return df


# Training and testing data
n_train = 1000
n_test = 100
theta_train = sample(theta_bar, radius, n=n_train)
theta_test = sample(theta_bar, radius, n=n_test)

m.train(theta_train, learner=mlopt.PYTORCH)

results = m.performance(theta_test)
print("Accuracy: %.2f " % results[0]['accuracy'])

# save training data
m.save_training_data("knapsack_training_data.pkl", delete_existing=True)

problem = cp.Problem(cp.Minimize(cost), constraints)
m = mlopt.Optimizer(problem)
m.load_training_data("knapsack_training_data.pkl")
m.train(learner=mlopt.PYTORCH)  # Train after loading samples

results = m.performance(theta_test)
print("Accuracy: %.2f " % results[0]['accuracy'])

# Predict single point
theta = theta_test.iloc[0]
result_single_point = m.solve(theta)
print(result_single_point)

y = m.y_train
X = m.X_train
learner = XGBoost(n_input=n_features(X),
                  n_classes=len(np.unique(y)),
                  n_best=3)
# Train learner
learner.train(pandas2array(X), y)

# Predict
X_pred = X.iloc[0]
y_pred = learner.predict(pandas2array(X_pred))  # n_best most likely classes