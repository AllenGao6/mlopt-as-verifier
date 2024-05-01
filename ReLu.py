# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 21:03:49 2024

@author: zhexi
"""

import cvxpy as cp
import numpy as np

# Generate a random problem
np.random.seed(0)
n = 5 # Number of neurons
layer = 5 # Number layers
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
constr += [x[:, 0] <= np.zeros(n) + 1, x[:, 0] >= np.zeros(n) - 1]

''' Specify output layer maximum'''
for j in range(n):
    constr += [x_out >= x[j, layer], x_out <= x[j, layer] + (1 - z_out[j]) * M]
constr += [cp.sum(z_out) == 1]

# Construct a CVXPY problem
objective = cp.Minimize(cp.norm(x_out, 1))
prob = cp.Problem(objective, constr)
prob.solve()

print("Status: ", prob.status)
print("The optimal value is", prob.value)
print("A solution x is")
print(x_out.value)