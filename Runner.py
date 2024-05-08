import numpy as np
import cvxpy as cp
import pandas as pd
import logging

import mlopt
from mlopt.sampling import uniform_sphere_sample
from mlopt.learners import XGBoost
from mlopt.utils import n_features, pandas2array

np.random.seed(1)  # Reset random seed for reproducibility

n = 2 # Number of neurons
layer = 4  # Number of layers
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
    return np.random.normal(loc=center, scale=radius, size=(n, len(center)))

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

m.train(theta_train, learner=mlopt.PYTORCH)


results = m.performance(theta_test)
print(len(results))
print("Accuracy: %.2f " % results[0]['accuracy'])

# Predict single point
# for i in range(5):
#     theta = theta_test.iloc[i]
#     result_single_point = m.solve(theta)
#     print(result_single_point)

'''
format of result_single_point:
'time': 0.0016522407531738281, 'strategy': <mlopt.strategy.Strategy object at 0x7fda8e7e3a90>, 'cost': 0.08093849178430029, 'infeasibility': 8.022732615307884e-05, 'pred_time': 0.001313924789428711, 'solve_time': 0.0003383159637451172}
'''
# get the average of the cost, time, infeasibility, pred_time, solve_time
cost = 0
time = 0
infeasibility = 0
pred_time = 0
solve_time = 0
ran = 5
for i in range(ran):
    theta = theta_test.iloc[i]
    result_single_point = m.solve(theta)
    print(result_single_point)
    cost += result_single_point['cost']
    time += result_single_point['time']
    infeasibility += result_single_point['infeasibility']
    pred_time += result_single_point['pred_time']
    solve_time += result_single_point['solve_time']
cost /= ran
time /= ran
infeasibility /= ran
pred_time /= ran
solve_time /= ran
print("Average cost: %.2f " % cost)
print("Average time: %.2f " % time)
print("Average infeasibility: %.2f " % infeasibility)
print("Average pred_time: %.2f " % pred_time)
print("Average solve_time: %.2f " % solve_time)

