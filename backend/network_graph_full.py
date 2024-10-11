import os
import time
import argparse
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
from cluster import find_mdl
from data import build_time_frames
from network_graph.data_transformation import clean_data, data_transformation

def load_and_transform_data(data_name):
    data = clean_data(data_name, 7)
    print("Starting data transformation...")
    start_time = time.time()

    ts_data = data_transformation(data)
    elapsed_time = time.time() - start_time
    print("Data transformation completed in", elapsed_time, "seconds")

    return data, ts_data

def run_algorithm(data_name, weight):
    data, ts_data = load_and_transform_data(data_name)

    print("Start building frames")
    tfs = build_time_frames(ts_data, 'time_value', ['value'])
    original_timeline_length = len(tfs)

    start_time = time.time()
    print("Start finding mdl")
    best_folded_timeline, min_dl = find_mdl(tfs, weight=weight)
    print("Done finding mdl")
    folded_timeline_length = len(best_folded_timeline)
    elapsed_time = time.time() - start_time
    print("MDL process runtime:", elapsed_time, "seconds")
    print("Original length:", original_timeline_length)
    print("Best folded timeline length:", folded_timeline_length)

    dates = [point.start_point.time_value for point in best_folded_timeline]
    dates.append(best_folded_timeline[-1].end_point.time_value)
    df = data[data['dep_time'].isin(dates) & data['arr_time'].isin(dates)]

    output_path = os.path.join('output/network_graph/', data_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    return data, df, output_path, elapsed_time, original_timeline_length, folded_timeline_length

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

# def draw_rotated_labels(G, pos, ax, distance_factor=1.2):
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

#         ax.text(label_x, label_y, str(node), size=12, rotation=rotation,
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

# def create_node_link_graph(df):
#     G = nx.DiGraph()
#     # Assuming df['dep_time'] and df['arr_time'] are your dataframe columns for departure and arrival times.
#     unique_time_values = pd.concat([df['dep_time'], df['arr_time']]).unique()
#     # Sort unique time values
#     unique_time_values = np.sort(unique_time_values)
#     for time in unique_time_values:
#         G.add_node(time)

#     count = 0
#     # Add edges with 'airline' attribute
#     for _, row in df.iterrows():
#         count += 1
#         G.add_edge(row['dep_time'], row['arr_time'], airline=row['name'])

#     # Determine the size of each node by the number of connections (both in and out)
#     node_sizes = [G.degree(node) * 10 for node in G.nodes()]

#     # Determine the alpha values for each node based on its degree
#     max_degree = max(G.degree(node) for node in G.nodes())
#     node_alphas = {node: 0.4 + 0.6 * (G.degree(node) / max_degree) for node in G.nodes()}  # Ensures alpha is between 0.4 and 1

#     # Create a circular layout
#     circle_pos = nx.circular_layout(G)

#     fig, ax = plt.subplots(figsize=(12, 12))

#     # Draw each node individually with its specific alpha
#     for node, size in zip(G.nodes(), node_sizes):
#         nx.draw_networkx_nodes(G, circle_pos, nodelist=[node], node_size=[size],
#                                node_color='blue', alpha=node_alphas[node], ax=ax)

#     # Remove labels from the graph and print them out
#     node_labels = {node: str(node) for node in G.nodes()}
#     print("Node Labels:")
#     for node_label in node_labels.values():
#         print(node_label)

#     # Custom function to draw edges with a gradient from blue to red
#     def draw_edge_with_gradient(u, v, pos, ax, cmap='coolwarm', lw=1):
#         spos = np.array(pos[u])
#         epos = np.array(pos[v])
#         svect = epos - spos
#         ctrl1 = spos + svect / 3 + np.array([-svect[1], svect[0]]) * 0.2
#         ctrl2 = spos + 2 * svect / 3 + np.array([-svect[1], svect[0]]) * 0.2

#         curve_path = Path([spos, ctrl1, ctrl2, epos], [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
#         patch = mpatches.PathPatch(curve_path, lw=lw, alpha=0.5, color='none', fill=False, ec='none')
#         ax.add_patch(patch)

#         gradient = np.linspace(0, 1, 100)
#         x = np.linspace(spos[0], epos[0], 100)
#         y = np.linspace(spos[1], epos[1], 100)
#         for s, e in zip(gradient[:-1], gradient[1:]):
#             color = plt.get_cmap(cmap)(s)
#             ax.plot(x[int(s*99):int(e*99)+1], y[int(s*99):int(e*99)+1], color=color, lw=lw)

#     for u, v in G.edges():
#         draw_edge_with_gradient(u, v, circle_pos, ax, lw=1)

#     plt.axis('off')
#     return plt

def visualize_results(data, filtered_data, output_path):
    # Visualize the initial graph
    plt_initial = create_node_link_graph(data)
    init_graph_path = os.path.join(output_path, "init.png")
    plt_initial.savefig(init_graph_path)
    plt_initial.close()
    
    # Visualize the folded graph
    plt_folded = create_node_link_graph(filtered_data)
    folded_graph_path = os.path.join(output_path, "folded.png")
    plt_folded.savefig(folded_graph_path)
    plt_folded.close()

    return init_graph_path, folded_graph_path

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process data and generate network graphs.')
    parser.add_argument('data_name', type=str, help='Name of the dataset')
    parser.add_argument('weight', type=float, help='Weight for the MDL algorithm')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    data, df, output_path, runtime, original_length, folded_length = run_algorithm(args.data_name, args.weight)
    print(f"Algorithm runtime: {runtime} seconds")
    print(f"Original timeline length: {original_length}")
    print(f"Folded timeline length: {folded_length}")
    init_graph_path, folded_graph_path = visualize_results(data, df, output_path)
    print(f"Initial graph saved at: {init_graph_path}")
    print(f"Folded graph saved at: {folded_graph_path}")