import cvxpy as cp
import numpy as np
import time

def solve_NN(n, layer, input_range, invariance = 1):
    M = 1e4
    x = cp.Variable((n, layer + 1))
    z_1 = cp.Variable((n, layer), boolean=True)
    z_2 = cp.Variable((n, layer), boolean=True)
    z_out = cp.Variable(n, boolean=True)
    x_out = cp.Variable(1)
    cost = 0
    constr = []

    for l in range(layer):
        W = np.random.rand(n, n) * invariance
        b = np.random.randn(n) * invariance

        for j in range(n):
            constr += [x[j, l + 1] >= W[j, :] @ x[:, l] + b[j],
                       x[j, l + 1] >= 0,
                       x[j, l + 1] <= W[j, :] @ x[:, l] + b[j] + (1 - z_1[j, l]) * M,
                       x[j, l + 1] <= (1 - z_2[j, l]) * M,
                       z_1[j, l] + z_2[j, l] == 1]

    constr += [x[:, 0] <= np.zeros(n) + input_range, x[:, 0] >= np.zeros(n) - input_range]

    for j in range(n):
        constr += [x_out >= x[j, layer], x_out <= x[j, layer] + (1 - z_out[j]) * M]
    constr += [cp.sum(z_out) == 1]

    objective = cp.Minimize(cp.norm(x_out, 1))
    prob = cp.Problem(objective, constr)

    start = time.time()
    prob.solve(verbose=False)
    end = time.time()
    return end - start, x_out.value

# Define parameter ranges
n_range = range(2, 90, 3)
layer_range = range(2, 15, 2)
input_range = range(1, 8, 2)

# Initialize dictionary to store results
results = {}

# Loop over all combinations of parameters
for n in n_range:
    for layer in layer_range:
        for input_val in input_range:
            print(f"Solving for n={n}, layer={layer}, input_range={input_val}")
            try:
                time_elapsed, optimal_value = solve_NN(n, layer, input_val)
            except Exception as e:
                print(f"Failed for n={n}, layer={layer}, input_range={input_val}")
                print(e)
                time_elapsed, optimal_value = -1, -1
            results[(n, layer, input_val)] = {'time_elapsed': time_elapsed, 'optimal_value': optimal_value}

# Save results to a file or process as needed
# For example, to save to a JSON file:
import json
with open('results.json', 'w') as f:
    json.dump(results, f)
