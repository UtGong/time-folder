import os
import time
import argparse
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
    print("done finding mdl")
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

    data.to_pickle('transformed_data.pkl')
    df.to_pickle('filtered_data.pkl')
    with open('meta_ng.txt', 'w') as f:
        f.write(f"{data_name},{weight},{output_path}")

    return data, df, output_path, elapsed_time, original_timeline_length, folded_timeline_length

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
