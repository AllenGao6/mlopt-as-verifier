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

    # Parameters for weights and biases
    W = cp.Parameter((n * layer, n), name='W')
    b = cp.Parameter((n, layer), name='b')

    cost = 0
    constr = []

    # Fill W and b with random values (as example, replace with real data in practice)
    W.value = np.random.rand(n * layer, n) * invariance
    b.value = np.random.randn(n, layer) * invariance

    for l in range(layer):
        for j in range(n):
            W_l = W[l*n:(l+1)*n, :]  # Extract the weight matrix for layer l
            b_l = b[j, l]  # Extract the bias for neuron j in layer l
            
            constr += [x[j, l + 1] >= W_l[j, :] @ x[:, l] + b_l,
                       x[j, l + 1] >= 0,
                       x[j, l + 1] <= W_l[j, :] @ x[:, l] + b_l + (1 - z_1[j, l]) * M,
                       x[j, l + 1] <= (1 - z_2[j, l]) * M,
                       z_1[j, l] + z_2[j, l] == 1]

    constr += [x[:, 0] <= np.zeros(n) + input_range, x[:, 0] >= np.zeros(n) - input_range]

    for j in range(n):
        constr += [x_out >= x[j, layer], x_out <= x[j, layer] + (1 - z_out[j]) * M]
    constr += [cp.sum(z_out) == 1]

    objective = cp.Minimize(cp.norm(x_out, 1))
    prob = cp.Problem(objective, constr)
    start = time.time()
    prob.solve()
    end = time.time()

    if not x_out or not x_out.value:
        return end - start, -1
    return end - start, float(x_out.value)

n_range = range(2, 45, 3)
layer_range = range(2, 9, 2)
input_range = range(1, 8, 2)

for n in n_range:
    for layer in layer_range:
        for input_val in input_range:
            print(f"Solving for n={n}, layer={layer}, input_range={input_val}")
            try:
                time_elapsed, optimal_value = solve_NN(n, layer, input_val)
                print(f"Time elapsed: {time_elapsed:.2f}s, Optimal value: {optimal_value}")
            except Exception as e:
                print(f"Failed for n={n}, layer={layer}, input_range={input_val}")
                print(e)
                time_elapsed, optimal_value = -1, -1
