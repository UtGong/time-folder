import os
import time
import argparse
from cluster import find_mdl
from data import build_time_frames
from network_graph.data_transformation import clean_data, create_node_link_graph, data_transformation

def main(data_name, weight):
    # Clean the data for the specified dataset name
    data = clean_data(data_name, 7)
    print("Starting data transformation...")
    start_time = time.time()
    
    # Perform data transformation
    ts_data = data_transformation(data)
    elapsed_time = time.time() - start_time
    print("Data transformation completed in", elapsed_time, "seconds")

    # Build time frames for analysis
    print("Start find mdl")
    tfs = build_time_frames(ts_data, 'time_value', ['value'])
    
    start_time = time.time()
    # Find the most descriptive timeline model using the given weight
    best_folded_timeline, min_dl = find_mdl(tfs, weight=weight)
    elapsed_time = time.time() - start_time
    print("MDL process runtime:", elapsed_time, "seconds")
    
    # Filter data for specific dates
    dates = [point.start_point.time_value for point in best_folded_timeline]
    dates.append(best_folded_timeline[-1].end_point.time_value)
    df = data[data['dep_time'].isin(dates) & data['arr_time'].isin(dates)]
    
    # Prepare the output directory
    output_path = os.path.join('output/network_graph/', data_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Create and save the initial graph
    G = create_node_link_graph(data)
    init_graph_path = os.path.join(output_path, "init.png")
    G.savefig(init_graph_path)
    
    # Create and save the graph after transformations
    G_merged = create_node_link_graph(df)
    folded_graph_path = os.path.join(output_path, "folded.png")
    G_merged.savefig(folded_graph_path)

    return init_graph_path, folded_graph_path

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process data and generate network graphs.')
    parser.add_argument('data_name', type=str, help='Name of the dataset')
    parser.add_argument('weight', type=float, help='Weight for the MDL algorithm')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    main(args.data_name, args.weight)