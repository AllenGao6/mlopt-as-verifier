import matplotlib.pyplot as plt
from mlopt.sampling import uniform_sphere_sample
import mlopt
import pandas as pd
import cvxpy as cp
import numpy as np
import os

np.random.seed(0)

np.random.seed(1)
T = 30
M = 3.
h = 1.
c = 2.
p = 3.

# Define problem
x = cp.Variable(T+1)
u = cp.Variable(T)

# Define parameters
d = cp.Parameter(T, nonneg=True, name="d")
x_init = cp.Parameter(1, name="x_init")

# Constaints
constraints = [x[0] == x_init]
for t in range(T):
    constraints += [x[t+1] == x[t] + u[t] - d[t]]
constraints += [u >= 0, u <= M]

# Objective
cost = cp.sum(cp.maximum(h * x, -p * x)) + c * cp.sum(u)

# Define optimizer
problem = cp.Problem(cp.Minimize(cost), constraints)
m = mlopt.Optimizer(problem)

# Average request
theta_bar = np.concatenate(( 2 * np.ones(T),  # d
                            [10]              # x_init
                           ))
radius = 1


def sample_inventory(theta_bar, radius, n=100):

    # Sample points from multivariate ball
    X_d = uniform_sphere_sample(theta_bar[:-1], radius, n=n)
    X_x_init = uniform_sphere_sample([theta_bar[-1]], 3 * radius,
                                     n=n)

    df = pd.DataFrame({'d': X_d.tolist(),
                       'x_init': X_x_init.tolist()})

    return df

# Training and testing data
n_train = 1000
n_test = 100
theta_train = sample_inventory(theta_bar, radius, n=n_train)
theta_test = sample_inventory(theta_bar, radius, n=n_test)

# Train solver
m.train(theta_train, learner=mlopt.PYTORCH)