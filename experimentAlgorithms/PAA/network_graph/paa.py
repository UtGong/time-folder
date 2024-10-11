import numpy as np
import pandas as pd
import time
from pyts.approximation import PiecewiseAggregateApproximation
from datetime import datetime, timedelta
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as mpatches

# Function to round time to the nearest multiple of a given number of minutes
def round_to_nearest(t, round_to_minute):
    if round_to_minute <= 0 or round_to_minute >= 60:
        raise ValueError("round_to_minute must be between 1 and 59")
    full_datetime = datetime(100, 1, 1, t.hour, t.minute)
    remainder = t.minute % round_to_minute
    if remainder == 0:
        pass
    elif remainder < round_to_minute / 2:
        full_datetime -= timedelta(minutes=remainder)
    else:
        full_datetime += timedelta(minutes=(round_to_minute - remainder))
    return full_datetime.time()

# Function to clean the data
def clean_data(duration):
    data = pd.read_csv("../../../backend/data/flights.csv")
    data = data.dropna()
    data['date'] = pd.to_datetime(data['year'].astype(str) + '-' + data['month'].astype(str) + '-' + data['day'].astype(str))
    data = data[(data['date'] >= data['date'].min()) & (data['date'] <= data['date'].min() + timedelta(days=duration))]
    data['dep_time'] = pd.to_datetime(data['dep_time'], format='%H%M', errors='coerce').dt.time
    data['arr_time'] = pd.to_datetime(data['arr_time'], format='%H%M', errors='coerce').dt.time
    data = data.dropna()
    data['dep_time'] = data['dep_time'].apply(lambda x: round_to_nearest(x, 10))
    data['arr_time'] = data['arr_time'].apply(lambda x: round_to_nearest(x, 10))
    data = data[['dep_time', 'arr_time', 'name']]
    return data

# Start time for measuring runtime
start_time = time.time()

# Load the dataset
data = pd.read_csv("../../../backend/data/analysis_df.csv")

# Extract time_value and ensure other data is numeric and handle missing values
time_values = data['time_value']
data = data.drop(columns=['time_value']).apply(pd.to_numeric, errors='coerce').fillna(0)

# Extract dep_count and arr_count as separate time series
dep_count_series = data['dep_count'].values
arr_count_series = data['arr_count'].values

# Combine these series into a 2D array for PAA
time_series_data = np.vstack([dep_count_series, arr_count_series])

# Define PAA parameters
window_size = 2
paa = PiecewiseAggregateApproximation(window_size=window_size)

# Perform PAA
compressed_data = paa.transform(time_series_data)

# Convert the compressed data back to a DataFrame for easier handling
compressed_df = pd.DataFrame(compressed_data.T, columns=['dep_count', 'arr_count'])

# Determine the original time_value entries corresponding to compressed data
compressed_indices = np.linspace(0, len(time_values) - 1, len(compressed_df), endpoint=True, dtype=int)
kept_time_values = time_values.iloc[compressed_indices].reset_index(drop=True)

# Combine kept time values with the compressed data
compressed_df['time_value'] = kept_time_values

# Reorder columns to have time_value first
compressed_df = compressed_df[['time_value', 'dep_count', 'arr_count']]

data = clean_data(7)

# Convert time_values to string to ensure compatibility during filtering
kept_time_values = kept_time_values.astype(str)

# Filter to keep the rows in data where dep_time and arr_time are in kept_time_values
filtered_data = data[data['dep_time'].astype(str).isin(kept_time_values) & data['arr_time'].astype(str).isin(kept_time_values)]

print('Original data shape:', time_series_data.shape)
print('Compressed data shape:', compressed_data.shape)
print(filtered_data)

# Measure and print runtime
run_time = time.time() - start_time
print('Runtime:', run_time)

# Visualization: Create a circular relationship graph
def draw_edge_with_gradient(u, v, pos, ax, cmap=plt.cm.coolwarm, lw=1):
    spos = np.array(pos[u])
    epos = np.array(pos[v])
    verts = [spos, epos]
    codes = [Path.MOVETO, Path.LINETO]
    path = Path(verts, codes)
    patch = mpatches.PathPatch(path, lw=lw, edgecolor='none')
    ax.add_patch(patch)

    # Calculate the gradient colors
    num_steps = 100  # Number of steps in the gradient
    for j in range(num_steps):
        color = cmap(j / num_steps)
        segment_start = spos + (epos - spos) * (j / num_steps)
        segment_end = spos + (epos - spos) * ((j + 1) / num_steps)
        ax.plot([segment_start[0], segment_end[0]], [segment_start[1], segment_end[1]], color=color, lw=lw)

# Function to rotate labels based on their position and adjust distance
# def draw_rotated_labels(G, pos, ax, distance_factor=1.1):
#     for node, (x, y) in pos.items():
#         angle = np.degrees(np.arctan2(y, x))
#         # Adjust the rotation angle for readability (flip for top vs bottom half)
#         if x > 0:
#             rotation = angle
#         else:
#             rotation = angle + 180

#         # Adjust label distance from the node
#         label_x = x * distance_factor
#         label_y = y * distance_factor

#         ax.text(label_x, label_y, str(node), size=8, rotation=rotation,
#                 rotation_mode='anchor', ha='center', va='center')

def draw_rotated_labels(G, pos, ax, distance_factor=1.2):
    for node, (x, y) in pos.items():
        # Only label nodes that represent whole hours (i.e., minutes and seconds are 0)
        # if node.minute == 0 and node.second == 0:
            # Format the label to show only 'HH:MM' in 24-hour format (e.g., '13:00')
        label = node.strftime("%H:%M")  # 24-hour format

        angle = np.degrees(np.arctan2(y, x))
        
        # Adjust the rotation angle for readability (flip for top vs bottom half)
        if x > 0:
            rotation = angle
        else:
            rotation = angle + 180

        # Adjust label distance from the node
        label_x = x * distance_factor
        label_y = y * distance_factor

        # Draw the label at the computed position with rotation
        ax.text(label_x, label_y, label, size=12, rotation=rotation,
                rotation_mode='anchor', ha='center', va='center')

# Create the directed graph
G = nx.DiGraph()

# Add nodes for each unique time
for time in pd.concat([filtered_data['dep_time'], filtered_data['arr_time']]).unique():
    G.add_node(time)

# Add edges for each flight
for _, row in filtered_data.iterrows():
    G.add_edge(row['dep_time'], row['arr_time'], airline=row['name'])

# Define node sizes based on degree (number of connections)
node_sizes = [G.degree(node) * 10 for node in G.nodes()]

# Define node colors and alpha (transparency) based on degree
max_degree = max(G.degree(node) for node in G.nodes())
node_alphas = {node: 0.4 + 0.6 * (G.degree(node) / max_degree) for node in G.nodes()}

# Create a circular layout for the graph
pos = nx.circular_layout(G)

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 10))

# Draw nodes with specific alpha values
for node, size in zip(G.nodes(), node_sizes):
    nx.draw_networkx_nodes(G, pos, nodelist=[node], node_size=[size],
                           node_color='blue', alpha=node_alphas[node], ax=ax)

# Draw rotated labels with increased distance
draw_rotated_labels(G, pos, ax, distance_factor=1.15)  # Adjust distance factor for label positioning

# Draw edges with gradient colors
for u, v, data in G.edges(data=True):
    draw_edge_with_gradient(u, v, pos, ax, cmap=plt.cm.coolwarm, lw=0.5)

# Remove axis
plt.axis('off')

# Set plot title
plt.title('PAA', y=1.05)

# Save the plot
plt.savefig("flight_network_graph.png")
plt.close()
