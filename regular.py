import numpy as np
import cvxpy as cp

# Reset random seed for reproducibility
np.random.seed(1)

n = 5  # Number of neurons
layer = 5  # Number of layers
M = 1e4

# Variables
x = cp.Variable((n, layer + 1))
z_1 = cp.Variable((n, layer), boolean=True)
z_2 = cp.Variable((n, layer), boolean=True)
z_out = cp.Variable(n, boolean=True)
x_out = cp.Variable(1)
constr = []

# Parameters for weights and biases, reshaping W to 2D
W = cp.Parameter((n * layer, n), name='W')  # Changed dimension of W
b = cp.Parameter((n, layer), name='b')

# Random weights and biases
random_W = np.random.rand(n * layer, n) * 2 - 1  # Random weights between -1 and 1
random_b = np.random.rand(n, layer) * 2 - 1  # Random biases between -1 and 1

# Set the values of W and b
W.value = random_W
b.value = random_b

for l in range(layer):
    for j in range(n):
        W_l = W[l*n:(l+1)*n, :]  # Extract weights for layer l
        constr += [x[j, l + 1] >= W_l[:, j] @ x[:, l] + b[j, l],
                   x[j, l + 1] >= 0,
                   x[j, l + 1] <= W_l[:, j] @ x[:, l] + b[j, l] + (1 - z_1[j, l]) * M,
                   x[j, l + 1] <= (1 - z_2[j, l]) * M,
                   z_1[j, l] + z_2[j, l] == 1]

# Specify input range
constr += [x[:, 0] <= np.zeros(n) + 1, x[:, 0] >= np.zeros(n) - 1]

# Specify output layer maximum
for j in range(n):
    constr += [x_out >= x[j, layer], x_out <= x[j, layer] + (1 - z_out[j]) * M]
constr += [cp.sum(z_out) == 1]

# Construct a CVXPY problem
objective = cp.Minimize(cp.norm(x_out, 1))
prob = cp.Problem(objective, constr)

# Solve the problem using the Gurobi solver
prob.solve(solver=cp.GUROBI, verbose=True)

print("Optimal value:", prob.value)
print("x_out value:", x_out.value)
