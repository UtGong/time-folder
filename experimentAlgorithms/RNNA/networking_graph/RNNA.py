from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import time
from scipy.interpolate import interp1d
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.path import Path
import matplotlib.patches as mpatches

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

# Combine these series into a 2D array for RAE
time_series_data = np.vstack([dep_count_series, arr_count_series])

# Load the trained RAE model
rae_model = load_model('rae_model.h5')

# Normalize the data
scaler = MinMaxScaler(feature_range=(-1, 1))
temps_scaled = scaler.fit_transform(time_series_data.T)

def interpolate(series, target_length):
    current_length = series.shape[0]
    if current_length == target_length:
        return series
    x = np.linspace(0, current_length - 1, current_length)
    f = interp1d(x, series, axis=0, kind='linear')
    x_new = np.linspace(0, current_length - 1, target_length)
    return f(x_new)

def adaptive_pairwise_compression(S, rae_model, max_deviation, rae_len):
    data_len = S.shape[0]
    st = 0
    compressed_segments = []

    while st < data_len:
        input_len = data_len - st
        len_stride = input_len // 2
        last_valid_len = 0

        while st + input_len <= data_len and len_stride > 1:
            xt = S[st:st + input_len]
            x_tilde = interpolate(xt, rae_len)
            x_tilde = x_tilde.reshape((1, rae_len, S.shape[1]))  # Reshape for model prediction
            x_hat = rae_model.predict(x_tilde)[0]
            max_dev = np.max(np.abs(xt - interpolate(x_hat, input_len)))

            if max_dev <= max_deviation:
                last_valid_len = input_len
                input_len += len_stride
            else:
                input_len -= len_stride

            len_stride //= 2

        xt = S[st:st + last_valid_len]
        x_tilde = interpolate(xt, rae_len)
        x_tilde = x_tilde.reshape((1, rae_len, S.shape[1]))  # Reshape for model prediction
        x_hat = rae_model.predict(x_tilde)[0]
        compressed_segments.append((st, last_valid_len, x_hat))  # Store the starting point, length, and compressed data
        st += last_valid_len

    return compressed_segments

# Set the parameters for compression
max_deviation = 1  # Adjusted max deviation for more lenient compression
rae_len = 20  # Adjusted length for the RAE model

compressed_segments = adaptive_pairwise_compression(temps_scaled, rae_model, max_deviation, rae_len)
compressed_series = [item[2] for item in compressed_segments]  # Extract only the compressed data

# Flatten the list of compressed data segments and scale back
compressed_series_flattened = np.concatenate(compressed_series, axis=0)
compressed_series_scaled = scaler.inverse_transform(compressed_series_flattened)

# Determine the original time_value entries corresponding to compressed data
compressed_indices = np.linspace(0, len(time_values) - 1, len(compressed_series_scaled), endpoint=True, dtype=int)
kept_time_values = time_values.iloc[compressed_indices].reset_index(drop=True)

# Load the original flights data for filtering
cleaned_data = clean_data(7)

# Ensure kept_time_values is of string type
kept_time_values = kept_time_values.astype(str)

# Filter the original data for dep_time and arr_time in kept_time_values
filtered_data = cleaned_data[
    cleaned_data['dep_time'].astype(str).isin(kept_time_values) &
    cleaned_data['arr_time'].astype(str).isin(kept_time_values)
]

print('Original data shape:', time_series_data.shape)
print('Compressed data shape:', compressed_series_scaled.shape)
print('Filtered data shape:', filtered_data.shape)

# Measure and print runtime
run_time = time.time() - start_time
print('Runtime:', run_time)

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

def create_node_link_graph(df):
    G = nx.DiGraph()
    unique_time_values = pd.concat([df['dep_time'], df['arr_time']]).unique()
    unique_time_values = np.sort(unique_time_values)

    for time in unique_time_values:
        G.add_node(time)

    for _, row in df.iterrows():
        G.add_edge(row['dep_time'], row['arr_time'], airline=row['name'])

    node_sizes = [G.degree(node) * 10 for node in G.nodes()]
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

    # Draw edges with gradient colors
    for u, v in G.edges():
        draw_edge_with_gradient(u, v, pos, ax, lw=1)

    plt.axis('off')
    return plt

# # Use the filtered_data obtained from the compression process
plot = create_node_link_graph(filtered_data)
plt.title('RNNA', y=1.05)
plot.savefig("RNNA_network_graph.png")
plt.close()