import cvxpy as cp
import numpy as np
import time
import json

def solve_NN(n, layer, input_range=1, invariance=1):
    M = 1e4
    x = cp.Variable((n, layer + 1))
    z_1 = cp.Variable((n, layer), boolean=True)
    z_2 = cp.Variable((n, layer), boolean=True)
    z_out = cp.Variable(n, boolean=True)
    x_out = cp.Variable(1)
    cost = 0
    constr = []

    for l in range(layer):
        W = np.random.uniform(low=-invariance, high=invariance, size=(n, n))
        b = np.random.uniform(low=-invariance, high=invariance, size=(n))

        for j in range(n):
            constr += [
                x[j, l + 1] >= W[j, :] @ x[:, l] + b[j],
                x[j, l + 1] >= 0,
                x[j, l + 1] <= W[j, :] @ x[:, l] + b[j] + (1 - z_1[j, l]) * M,
                x[j, l + 1] <= (1 - z_2[j, l]) * M,
                z_1[j, l] + z_2[j, l] == 1
            ]

    constr += [x[:, 0] <= np.zeros(n) + input_range, x[:, 0] >= np.zeros(n) - input_range]

    for j in range(n):
        constr += [x_out >= x[j, layer], x_out <= x[j, layer] + (1 - z_out[j]) * M]
    constr += [cp.sum(z_out) == 1]

    objective = cp.Minimize(cp.norm(x_out, 1))
    prob = cp.Problem(objective, constr)

    start = time.time()
    prob.solve(verbose=False, solver=cp.GUROBI)
    end = time.time()
    time_elapsed = end - start

    if not x_out or not x_out.value:
        return time_elapsed, -1, -1
    return time_elapsed, float(x_out.value), prob.value

# Define parameter ranges
n_range = range(2, 28, 2)
layer_range = range(2, 6)
input_range = range(1, 6)

# Initialize dictionary to store results
results = {}

# Loop over all combinations of parameters
for n in n_range:
    for layer in layer_range:
        for input_val in input_range:
            print(f"Solving for n={n}, layer={layer}, input_range={input_val}")
            try:
                time_elapsed, optimal_value, cost = solve_NN(n, layer, input_val)
            except Exception as e:
                print(f"Failed for n={n}, layer={layer}, input_range={input_val}")
                print(e)
                time_elapsed, optimal_value, cost = -1, -1, -1
            key = str(n) + '_' + str(layer) + '_' + str(input_val)
            results[key] = {
                'time_elapsed': time_elapsed,
                'optimal_value': optimal_value,
                'cost': cost
            }

# Save results to a JSON file
with open('results.json', 'w') as f:
    json.dump(results, f, indent=4)
