# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 21:03:49 2024

@author: zhexi
"""

import cvxpy as cp
import numpy as np
import time

# Generate a random problem
np.random.seed(0)
n = 20 # Number of neurons
layer = 5 # Number layers

def NN(n, layer, input_range=1):
    M = 1e4
    x = cp.Variable((n, layer + 1))
    z_1 = cp.Variable((n, layer), boolean = True)
    z_2 = cp.Variable((n, layer), boolean = True)
    z_out = cp.Variable(n, boolean = True)
    x_out = cp.Variable(1)
    cost = 0
    constr = []

    for l in range(layer):
        W = np.random.rand(n, n)
        b = np.random.randn(n)
        
        for j in range(n):
            ''' Encoding ReLu activation function '''
            constr += [x[j, l + 1] >= W[j,:] @ x[:, l] + b[j], \
                    x[j, l + 1] >=0, \
                    x[j, l + 1] <= W[j,:] @ x[:, l] + b[j] + (1 - z_1[j, l]) * M, \
                    x[j, l + 1] <= (1 - z_2[j, l]) * M, \
                    z_1[j, l] + z_2[j, l] == 1]

    ''' Specify input range '''

    constr += [x[:, 0] <= np.zeros(n) + input_range, x[:, 0] >= np.zeros(n) - input_range]

    ''' Specify output layer maximum'''
    for j in range(n):
        constr += [x_out >= x[j, layer], x_out <= x[j, layer] + (1 - z_out[j]) * M]
    constr += [cp.sum(z_out) == 1]

    # Construct a CVXPY problem
    objective = cp.Minimize(cp.norm(x_out, 1))
    prob = cp.Problem(objective, constr)

    start = time.time()
    prob.solve(verbose=True)
    end = time.time()
    print("Time elapsed: ", end - start)
    return end - start, x_out.value

    # print("Status: ", prob.status)
    # print("The optimal value is", prob.value)
    # print("A solution x is")
    # print(x_out.value)

# Define parameter ranges
n_range = range(2, 101)
layer_range = range(2, 11)
input_range = range(1, 11)

# Initialize dictionary to store results
results = {}

# Loop over all combinations of parameters
for n in n_range:
    for layer in layer_range:
        for input_val in input_range:
            print(f"Solving for n={n}, layer={layer}, input_range={input_val}")
            time_elapsed, optimal_value = solve_NN(n, layer, input_val)
            results[(n, layer, input_val)] = {'time_elapsed': time_elapsed, 'optimal_value': optimal_value}

# Save results to a file or process as needed
# For example, to save to a JSON file:
import json
with open('results.json', 'w') as f:
    json.dump(results, f)