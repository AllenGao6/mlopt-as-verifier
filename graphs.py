import json
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

# Load data from JSON file
with open('results.json', 'r') as file:
    results = json.load(file)

# Initialize data structures
costs = defaultdict(list)
times = defaultdict(list)

# Process the results, filtering out data with value -1
for key, value in results.items():
    if value['optimal_value'] != -1 and value['cost'] != -1:
        key_prefix = '_'.join(key.split('_')[:2])
        costs[key_prefix].append(value['cost'])
        times[key_prefix].append(value['time_elapsed'])

# Calculate averages
avg_costs = {key: np.mean(vals) for key, vals in costs.items()}
avg_times = {key: np.mean(vals) for key, vals in times.items()}

# Organize data for plotting
neurons_layers = sorted(avg_costs.keys(), key=lambda s: (int(s.split('_')[0]), int(s.split('_')[1])))
neurons, layers = zip(*[(int(k.split('_')[0]), int(k.split('_')[1])) for k in neurons_layers])

# Get unique neurons and layers
unique_neurons = sorted(set(neurons))
unique_layers = sorted(set(layers))

# Prepare data for line graphs
def prepare_line_data(data_dict):
    line_data = {layer: [] for layer in unique_layers}
    for neuron in unique_neurons:
        for layer in unique_layers:
            key = f'{neuron}_{layer}'
            if key in data_dict:
                line_data[layer].append(data_dict[key])
            else:
                line_data[layer].append(np.nan)
    return line_data

cost_data = prepare_line_data(avg_costs)
time_data = prepare_line_data(avg_times)

# Plot Cost graph
plt.figure(figsize=(10, 5))
for layer, data in cost_data.items():
    plt.plot(unique_neurons, data, marker='o', label=f'Layer {layer}')
plt.xlabel('Number of Neurons')
plt.ylabel('Average Cost')
plt.title('Average Cost for Different Neuron Counts and Layers')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('average_cost_line.png')
plt.show()

# Plot Time graph
plt.figure(figsize=(10, 5))
for layer, data in time_data.items():
    plt.plot(unique_neurons, data, marker='o', label=f'Layer {layer}')
plt.xlabel('Number of Neurons')
plt.ylabel('Average Time Elapsed (seconds)')
plt.title('Average Time Elapsed for Different Neuron Counts and Layers')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('average_time_line.png')
plt.show()
