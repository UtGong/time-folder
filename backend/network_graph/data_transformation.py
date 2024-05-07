import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
from datetime import datetime, timedelta

def round_to_nearest(t, round_to_minute):
    # Ensure round_to_minute is valid
    if round_to_minute <= 0 or round_to_minute >= 60:
        raise ValueError("round_to_minute must be between 1 and 59")
    
    # Convert time object to a full datetime object
    full_datetime = datetime(100, 1, 1, t.hour, t.minute)
    
    # Calculate how many minutes to add or subtract to round to the nearest round_to_minute
    remainder = t.minute % round_to_minute
    if remainder == 0:
        # Time is already rounded
        pass
    elif remainder < round_to_minute / 2:
        # Subtract remainder minutes to round down
        full_datetime -= timedelta(minutes=remainder)
    else:
        # Add minutes to round up
        full_datetime += timedelta(minutes=(round_to_minute - remainder))
    
    return full_datetime.time()

def clean_data(file_name, duration):
    data = pd.read_csv("data/" + file_name + ".csv")
    data = data.dropna()

    # column year + month + day to date
    data['date'] = pd.to_datetime(data['year'].astype(str) + '-' + data['month'].astype(str) + '-' + data['day'].astype(str))
    
    # filter to keep those between the first day of date column and the duration
    data = data[(data['date'] >= data['date'].min()) & (data['date'] <= data['date'].min() + timedelta(days=duration))]
    
    # data = data[(data['date'] >= '2013-01-01') & (data['date'] <= '2013-01-06')]

    # Convert to datetime to parse the time, then round to nearest 10 minutes
    data['dep_time'] = pd.to_datetime(data['dep_time'], format='%H%M', errors='coerce').dt.time
    data['arr_time'] = pd.to_datetime(data['arr_time'], format='%H%M', errors='coerce').dt.time

    data = data.dropna()
    # Apply rounding function
    data['dep_time'] = data['dep_time'].apply(lambda x: round_to_nearest(x, 10))
    data['arr_time'] = data['arr_time'].apply(lambda x: round_to_nearest(x, 10))
    
    # keep only the dep_time, arr_time, name
    data = data[['dep_time', 'arr_time', 'name']]
    
    return data

# def create_node_link_graph(df):
#     G = nx.DiGraph()
#     unique_time_values = pd.concat([df['dep_time'], df['arr_time']]).unique()
#     # sort unique time values 
#     unique_time_values = np.sort(unique_time_values)
#     print("n time values:", len(unique_time_values))
#     for time in unique_time_values:
#         G.add_node(time)
    
#     count = 0
#     # Add edges with 'airline' attribute
#     for _, row in df.iterrows():
#         count += 1
#         G.add_edge(row['dep_time'], row['arr_time'], airline=row['name'])
    
#     print(count)
    
#     # Determine the size of each node by the number of connections (both in and out)
#     node_sizes = [G.degree(node) * 10 for node in G.nodes()]

#     # Generate a color map for edges based on unique airline names
#     unique_airlines = df['name'].unique()
#     colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_airlines)))
#     airline_color_map = dict(zip(unique_airlines, colors))

#     # Determine the color for each edge
#     edge_colors = [airline_color_map[G[u][v]['airline']] for u, v in G.edges()]

#     # Create a circular layout
#     circle_pos = nx.circular_layout(G)
    
#     fig, ax = plt.subplots(figsize=(12, 12))

#     # Draw nodes
#     nx.draw_networkx_nodes(G, circle_pos, node_size=node_sizes, node_color='lightblue', alpha=0.9, ax=ax)

#     # Draw labels
#     nx.draw_networkx_labels(G, circle_pos, font_size=8, ax=ax)

#     # Draw edges with Bezier curves
#     for (u, v), color in zip(G.edges(), edge_colors):
#         spos = np.array(circle_pos[u])
#         epos = np.array(circle_pos[v])
#         svect = epos - spos
#         # Control points for the Bezier curve, adjust the multiplier for the curvature
#         ctrl1 = spos + svect / 3 + np.array([-svect[1], svect[0]]) * 0.2
#         ctrl2 = spos + 2 * svect / 3 + np.array([-svect[1], svect[0]]) * 0.2
        
#         curve_path = Path([spos, ctrl1, ctrl2, epos], [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
        
#         patch = mpatches.PathPatch(curve_path, lw=1, alpha=0.5, color=color, fill=False)
#         ax.add_patch(patch)

#     # Remove axes
#     plt.axis('off')

#     return plt

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.path import Path

def create_node_link_graph(df):
    G = nx.DiGraph()
    # Assuming df['dep_time'] and df['arr_time'] are your dataframe columns for departure and arrival times.
    unique_time_values = pd.concat([df['dep_time'], df['arr_time']]).unique()
    # Sort unique time values
    unique_time_values = np.sort(unique_time_values)
    print("n time values:", len(unique_time_values))
    for time in unique_time_values:
        G.add_node(time)

    count = 0
    # Add edges with 'airline' attribute
    for _, row in df.iterrows():
        count += 1
        G.add_edge(row['dep_time'], row['arr_time'], airline=row['name'])
    
    print(count)
    
    # Determine the size of each node by the number of connections (both in and out)
    node_sizes = [G.degree(node) * 10 for node in G.nodes()]

    # Determine the alpha values for each node based on its degree
    max_degree = max(G.degree(node) for node in G.nodes())
    node_alphas = {node: 0.4 + 0.6 * (G.degree(node) / max_degree) for node in G.nodes()}  # Ensures alpha is between 0.4 and 1

    # Create a circular layout
    circle_pos = nx.circular_layout(G)
    
    fig, ax = plt.subplots(figsize=(12, 12))

    # Draw each node individually with its specific alpha
    for node, size in zip(G.nodes(), node_sizes):
        nx.draw_networkx_nodes(G, circle_pos, nodelist=[node], node_size=[size],
                               node_color='blue', alpha=node_alphas[node], ax=ax)

    # Draw labels
    nx.draw_networkx_labels(G, circle_pos, font_size=8, ax=ax)

    # Custom function to draw edges with a gradient from blue to red
    def draw_edge_with_gradient(u, v, pos, ax, cmap='coolwarm', lw=1):
        spos = np.array(pos[u])
        epos = np.array(pos[v])
        svect = epos - spos
        ctrl1 = spos + svect / 3 + np.array([-svect[1], svect[0]]) * 0.2
        ctrl2 = spos + 2 * svect / 3 + np.array([-svect[1], svect[0]]) * 0.2
        
        curve_path = Path([spos, ctrl1, ctrl2, epos], [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
        patch = mpatches.PathPatch(curve_path, lw=lw, alpha=0.5, color='none', fill=False, ec='none')
        ax.add_patch(patch)
        
        gradient = np.linspace(0, 1, 100)
        x = np.linspace(spos[0], epos[0], 100)
        y = np.linspace(spos[1], epos[1], 100)
        for s, e in zip(gradient[:-1], gradient[1:]):
            color = plt.get_cmap(cmap)(s)
            ax.plot(x[int(s*99):int(e*99)+1], y[int(s*99):int(e*99)+1], color=color, lw=lw)

    for u, v in G.edges():
        draw_edge_with_gradient(u, v, circle_pos, ax, lw=1)

    plt.axis('off')
    return plt

def data_transformation(df):
    # Getting all unique times
    unique_times = pd.Index(df['dep_time']).union(df['arr_time']).unique()
    
    # Preparing the final DataFrame
    analysis_df = pd.DataFrame({'time_value': unique_times})

    # Adding departure and arrival counts
    analysis_df['dep_count'] = analysis_df['time_value'].apply(lambda x: sum(df['dep_time'] == x))
    analysis_df['arr_count'] = analysis_df['time_value'].apply(lambda x: sum(df['arr_time'] == x))

    analysis_df.sort_values('time_value').reset_index(drop=True)
    
    # make a new column time_value where timevalue = dep_count * 2(weight) + arr_count
    analysis_df['value'] = analysis_df['dep_count'] + analysis_df['arr_count'] 
    analysis_df['dep_value'] = analysis_df['dep_count'] / analysis_df['value']
    analysis_df['arr_value'] = analysis_df['arr_count'] / analysis_df['value']
    # save analysis_df to csv
    analysis_df.to_csv("data/analysis_df.csv", index=False)
    return analysis_df



