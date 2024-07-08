import os
import time
import pandas as pd
import argparse
from data import build_time_frames
from cluster import find_mdl
from symbolic_seq.data_transformation import transform_HR_data

def load_and_transform_data(data_name):
    print("Starting data transformation...")
    start_time = time.time()
    df, unique_id = transform_HR_data()  # Transform HR data (assumed to be a placeholder function)
    elapsed_time = time.time() - start_time
    print("Done transforming data...")
    print("Runtime: ", elapsed_time)

    output_path = os.path.join('output/symbolic_seq/', data_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    return df, output_path

def run_algorithm(data_name, weight):
    df, output_path = load_and_transform_data(data_name)

    data_file_name = 'ss_to_ts'
    csv_file_path = os.path.join('./data', data_file_name + '.csv')
    date_column = 'time_value'
    value_column = ["tp_value"]

    dataset = pd.read_csv(csv_file_path)
    print("len(original)", len(dataset))

    start_time = time.time()
    print("Start find mdl process...")
    tfs = build_time_frames(dataset, date_column, value_column)
    original_timeline_length = len(tfs)
    best_folded_timeline, min_dl = find_mdl(tfs, weight=weight)
    folded_timeline_length = len(best_folded_timeline)
    elapsed_time = time.time() - start_time
    print("MDL process runtime:", elapsed_time, "seconds")

    dates = [point.start_point.time_value for point in best_folded_timeline]
    dates.append(best_folded_timeline[-1].end_point.time_value)

    filtered_df = df[df['time_value'].isin(dates)]
    print(f"Filtered data length: {len(filtered_df)}")

    df.to_pickle('transformed_data.pkl')
    filtered_df.to_pickle('filtered_data.pkl')
    with open('meta_ss.txt', 'w') as f:
        f.write(f"{data_name},{weight},{output_path}")

    return df, filtered_df, output_path, elapsed_time, original_timeline_length, folded_timeline_length

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('data_name', type=str, help='Name of the data set')
    parser.add_argument('weight', type=float, help='Weight parameter for the MDL process')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    df, filtered_df, output_path, runtime, original_length, folded_length = run_algorithm(args.data_name, args.weight)
    print(f"Algorithm runtime: {runtime} seconds")
    print(f"Original timeline length: {original_length}")
    print(f"Folded timeline length: {folded_length}")
