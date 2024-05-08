import numpy as np
import cvxpy as cp
import pandas as pd
import logging

import mlopt
from mlopt.sampling import uniform_sphere_sample
from mlopt.learners import XGBoost
from mlopt.utils import n_features, pandas2array

np.random.seed(1)  # Reset random seed for reproducibility

n = 20  # Number of neurons
layer = 8  # Number of layers
M = 1e4
x = cp.Variable((n, layer + 1))
z_1 = cp.Variable((n, layer), boolean=True)
z_2 = cp.Variable((n, layer), boolean=True)
z_out = cp.Variable(n, boolean=True)
x_out = cp.Variable(1)

# Parameters for weights and biases
W = cp.Parameter((n * layer, n), name='W')
b = cp.Parameter((n, layer), name='b')

cost = 0
constr = []

# Fill W and b with random values (as example, replace with real data in practice)

for l in range(layer):
    for j in range(n):
        W_l = W[l*n:(l+1)*n, :]  # Extract the weight matrix for layer l
        b_l = b[j, l]  # Extract the bias for neuron j in layer l
        
        constr += [x[j, l + 1] >= W_l[j, :] @ x[:, l] + b_l,
                x[j, l + 1] >= 0,
                x[j, l + 1] <= W_l[j, :] @ x[:, l] + b_l + (1 - z_1[j, l]) * M,
                x[j, l + 1] <= (1 - z_2[j, l]) * M,
                z_1[j, l] + z_2[j, l] == 1]
input_range = 1
constr += [x[:, 0] <= np.zeros(n) + input_range, x[:, 0] >= np.zeros(n) - input_range]

for j in range(n):
    constr += [x_out >= x[j, layer], x_out <= x[j, layer] + (1 - z_out[j]) * M]
constr += [cp.sum(z_out) == 1]

objective = cp.Minimize(cp.norm(x_out, 1))
prob = cp.Problem(objective, constr)

m = mlopt.Optimizer(prob, log_level=logging.INFO)



# Average request
theta_bar = np.random.rand(n * n * layer + n * layer)
radius = 1.0

def uniform_sphere_sample(center, radius, n=100):
    dim = center.shape[0]
    x = np.random.normal(size=(n, dim))
    norms = np.linalg.norm(x, axis=1)
    x_normalized = x / norms[:, np.newaxis]  # Normalize each sample to have norm 1
    x_sphere = center + radius * x_normalized  # Scale by radius and shift by center
    return x_sphere

def sample(theta_bar, radius, n_samples=100, n=10, layer=3):
    # Calculate lengths for slicing
    length_W = n * n * layer
    length_b = n * layer
    
    # Separate the theta_bar into parts for W and b
    theta_bar_W = theta_bar[:length_W]
    theta_bar_b = theta_bar[length_W:length_W + length_b]
    
    # Sample points from multivariate ball
    X_W = uniform_sphere_sample(theta_bar_W, radius, n=n_samples).reshape(n_samples, n * layer, n)
    X_b = uniform_sphere_sample(theta_bar_b, radius, n=n_samples).reshape(n_samples, n, layer)

    # Convert arrays to list of matrices for easier handling in DataFrames
    df = pd.DataFrame({
        'W': [X_W[i] for i in range(n_samples)],
        'b': [X_b[i] for i in range(n_samples)]
    })
    return df



# Training and testing data
n_train = 1000
n_test = 100
theta_train = sample(theta_bar, radius, n_samples=n_train)
theta_test = sample(theta_bar, radius, n_samples=n_test)

m.train(theta_train, learner=mlopt.OPTIMAL_TREE)


results = m.performance(theta_test)
print("Accuracy: %.2f " % results[0]['accuracy'])

# Predict single point
theta = theta_test.iloc[0]
result_single_point = m.solve(theta)
print(result_single_point)
