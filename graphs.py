import json
import matplotlib.pyplot as plt
import re

# Load data from results.json
with open('results.json', 'r') as file:
    data = json.load(file)

# Function to parse the key and extract the number of neurons and layer
def parse_key(key):
    match = re.match(r'(\d+)_(\d+)_\d+', key)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

# Create a dictionary to store optimal values
optimal_values = {}

# Parse the data
for key, value in data.items():
    neurons, layer = parse_key(key)
    if neurons and layer:
        if layer not in optimal_values:
            optimal_values[layer] = {}
        optimal_values[layer][neurons] = value['time_elapsed']

# Plot each layer's results
plt.figure(figsize=(10, 6))
for layer, values in sorted(optimal_values.items()):
    x = sorted(values.keys())
    y = [values[n] for n in x]
    plt.plot(x, y, marker='o', label=f'Layer {layer}')

# Add labels and legend
plt.xlabel('Number of Neurons')
plt.ylabel('Time Cost (seconds)')
plt.title('Time Cost for Different Neuron Counts and Layers')
plt.legend(title='Layers')
plt.grid(True)

# Show the plot
plt.show()
