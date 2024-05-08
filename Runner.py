import numpy as np
import cvxpy as cp
import pandas as pd
import logging

import mlopt
from mlopt.sampling import uniform_sphere_sample
from mlopt.learners import XGBoost
from mlopt.utils import n_features, pandas2array

np.random.seed(1)  # Reset random seed for reproducibility

n = 6 # Number of neurons
layer = 5  # Number of layers
M = 1e4

x = cp.Variable((n, layer + 1))
z_1 = cp.Variable((n, layer), boolean=True)
z_2 = cp.Variable((n, layer), boolean=True)
z_out = cp.Variable(n, boolean=True)
x_out = cp.Variable(1)
constr = []

# Parameters for weights and biases, reshaping W to 2D
W = cp.Parameter((n * layer, n), name='W')  # Changed dimension of W
b = cp.Parameter((n, layer), name='b')

for l in range(layer):
    for j in range(n):
        ''' Encoding ReLu activation function '''
        # Update matrix multiplication to align with the reshaped W
        W_l = W[l*n:(l+1)*n, :]  # Extract weights for layer l
        constr += [x[j, l + 1] >= W_l[:, j] @ x[:, l] + b[j, l],
                   x[j, l + 1] >= 0,
                   x[j, l + 1] <= W_l[:, j] @ x[:, l] + b[j, l] + (1 - z_1[j, l]) * M,
                   x[j, l + 1] <= (1 - z_2[j, l]) * M,
                   z_1[j, l] + z_2[j, l] == 1]

''' Specify input range '''
constr += [x[:, 0] <= np.zeros(n) + 1, x[:, 0] >= np.zeros(n) - 1]

''' Specify output layer maximum'''
for j in range(n):
    constr += [x_out >= x[j, layer], x_out <= x[j, layer] + (1 - z_out[j]) * M]
constr += [cp.sum(z_out) == 1]

# Construct a CVXPY problem
objective = cp.Minimize(cp.norm(x_out, 1))
prob = cp.Problem(objective, constr)
m = mlopt.Optimizer(prob, log_level=logging.INFO)



# Average request
theta_bar = np.ones(n * n * layer + n * layer) - 1
radius = 1.0

def uniform_sphere_sample(center, radius, n=100):
    # Simplified sampler, replace with actual sampling logic
    a = np.random.normal(loc=center, scale=radius, size=(n, len(center)))

    print(a)
    return a

def sample(theta_bar, radius, n_samples=100):
    # Calculate lengths for slicing
    length_W = n * n * layer
    length_b = n * layer
    
    # Sample points from multivariate ball
    X_W = uniform_sphere_sample(theta_bar[:length_W], radius, n=n_samples).reshape(n_samples, n * layer, n)
    X_b = uniform_sphere_sample(theta_bar[length_W:length_W + length_b], radius, n=n_samples).reshape(n_samples, n, layer)

    df = pd.DataFrame({
        'W': list(X_W),
        'b': list(X_b)
    })
    return df



# Training and testing data
n_train = 1000
n_test = 100
theta_train = sample(theta_bar, radius, n_samples=n_train)
theta_test = sample(theta_bar, radius, n_samples=n_test)

m.train(theta_train, learner=mlopt.XGBOOST)


results = m.performance(theta_test)
print("Accuracy: %.2f " % results[0]['accuracy'])

# Predict single point
theta = theta_test.iloc[0]
result_single_point = m.solve(theta)
print(result_single_point)