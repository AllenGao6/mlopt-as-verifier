import numpy as np
import cvxpy as cp
import pandas as pd
import logging
import mlopt

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
W = cp.Parameter((n * n * layer, n), name='W')
b = cp.Parameter((n * layer, 1), name='b')

for l in range(layer):
    for j in range(n):
        W_slice = W[(l * n * n + j * n):(l * n * n + (j + 1) * n), :]  # Extract the weights for neuron j in layer l
        b_slice = b[l * n + j]  # Bias for neuron j in layer l
        constr += [x[j, l + 1] >= W_slice @ x[:, l] + b_slice,
                   x[j, l + 1] >= 0,
                   x[j, l + 1] <= W_slice @ x[:, l] + b_slice + (1 - z_1[j, l]) * M,
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

# Initialize theta_bar based on flattened parameters
size_W = n * n * layer
size_b = n * layer
theta_bar = 2 * np.ones(size_W + size_b)  # Adjust dimensions for weights and biases
radius = 1.0

def uniform_sphere_sample(center, radius, n=100):
    return np.random.normal(loc=center, scale=radius, size=(n, len(center)))

def sample(theta_bar, radius, n=100):
    # Sample points from multivariate ball
    size_W = n * n * layer
    size_b = n * layer
    X_W = uniform_sphere_sample(theta_bar[:size_W], radius, n=n)
    X_b = uniform_sphere_sample(theta_bar[size_W:size_W + size_b], radius, n=n)

    df = pd.DataFrame({
        'W': list(X_W),
        'b': list(X_b)
    })
    return df

# Training and testing data
n_train = 1000
n_test = 100
theta_train = sample(theta_bar, radius, n=n_train)
theta_test = sample(theta_bar, radius, n=n_test)

m.train(theta_train, learner=mlopt.XGBOOST)

# Performance and single point prediction
results = m.performance(theta_test)
print("Accuracy: %.2f " % results[0]['accuracy'])

theta = theta_test.iloc[0]
result_single_point = m.solve(theta)
print(result_single_point)
