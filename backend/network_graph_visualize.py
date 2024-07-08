import os
import pandas as pd
from network_graph.data_transformation import create_node_link_graph

def visualize_results():
    data = pd.read_pickle('transformed_data.pkl')
    filtered_data = pd.read_pickle('filtered_data.pkl')
    
    with open('meta_ng.txt', 'r') as f:
        meta = f.read().split(',')
    data_name = meta[0]
    weight = float(meta[1])
    output_path = meta[2]

    G = create_node_link_graph(data)
    init_graph_path = os.path.join(output_path, "init.png")
    G.savefig(init_graph_path)

    G_merged = create_node_link_graph(filtered_data)
    folded_graph_path = os.path.join(output_path, "folded.png")
    G_merged.savefig(folded_graph_path)

    return init_graph_path, folded_graph_path

if __name__ == "__main__":
    init_graph_path, folded_graph_path = visualize_results()
    print(f"Initial graph saved at: {init_graph_path}")
    print(f"Folded graph saved at: {folded_graph_path}")
