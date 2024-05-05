import numpy as np
import cvxpy as cp
import pandas as pd
import logging
import mlopt
from mlopt.sampling import uniform_sphere_sample
from mlopt.learners import XGBoost
from mlopt.utils import n_features, pandas2array

np.random.seed(1)  # Reset random seed for reproducibility

n = 5  # Number of neurons
layer = 5  # Number layers
M = 1e4

x = cp.Variable((n, layer + 1))
z_1 = cp.Variable((n, layer), boolean=True)
z_2 = cp.Variable((n, layer), boolean=True)
z_out = cp.Variable(n, boolean=True)
x_out = cp.Variable(1)

constr = []

# Parameters for weights and biases
W = [cp.Parameter((n, n), name=f'W_{l}') for l in range(layer)]
b = [cp.Parameter(n, name=f'b_{l}') for l in range(layer)]

for l in range(layer):
    for j in range(n):
        ''' Encoding ReLu activation function '''
        constr += [x[j, l + 1] >= W[l][:, j] @ x[:, l] + b[l][j],
                   x[j, l + 1] >= 0,
                   x[j, l + 1] <= W[l][:, j] @ x[:, l] + b[l][j] + (1 - z_1[j, l]) * M,
                   x[j, l + 1] <= (1 - z_2[j, l]) * M,
                   z_1[j, l] + z_2[j, l] == 1]

''' Specify input range '''
constr += [x[:, 0] <= np.zeros(n) + 1,
           x[:, 0] >= np.zeros(n) - 1]

''' Specify output layer maximum'''
for j in range(n):
    constr += [x_out >= x[j, layer],
               x_out <= x[j, layer] + (1 - z_out[j]) * M]

constr += [cp.sum(z_out) == 1]

# Construct a CVXPY problem
objective = cp.Minimize(cp.norm(x_out, 1))
prob = cp.Problem(objective, constr)

m = mlopt.Optimizer(prob, log_level=logging.INFO)

# Average request
theta_bar = 2 * np.ones(n * layer * n + n * layer)  # Adjust dimensions for weights and biases
radius = 1.0

def uniform_sphere_sample(center, radius, n=100):
    # Simplified sampler, replace with actual sampling logic
    return np.random.normal(loc=center, scale=radius, size=(n, len(center)))

def sample(theta_bar, radius, n=100):
    # Sample points from multivariate ball
    X_W = [uniform_sphere_sample(theta_bar[l*n*n:(l+1)*n*n], radius, n=n).reshape(n, n, n) for l in range(layer)]
    X_b = [uniform_sphere_sample(theta_bar[n*n*layer+l*n:n*n*layer+(l+1)*n], radius, n=n).reshape(n, n) for l in range(layer)]
    df = pd.DataFrame({f'W_{l}': list(X_W[l]) for l in range(layer)} | {f'b_{l}': list(X_b[l]) for l in range(layer)})
    return df

# Training and testing data
n_train = 1000
n_test = 100
theta_train = sample(theta_bar, radius, n=n_train)
theta_test = sample(theta_bar, radius, n=n_test)

m.train(theta_train, learner=mlopt.XGBOOST)
results = m.performance(theta_test)
print("Accuracy: %.2f " % results[0]['accuracy'])

# Predict single point
theta = theta_test.iloc[0]
result_single_point = m.solve(theta)
print(result_single_point)