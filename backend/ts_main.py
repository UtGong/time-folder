# import os
# import time
# import pandas as pd
# import argparse
# from cluster_ts import find_mdl
# from data import build_time_frames

# def load_and_filter_data(csv_file_path, date_range, date_column, value_column):
#     csv = pd.read_csv(csv_file_path)
#     csv.dropna(inplace=True)
#     csv[date_column] = pd.to_datetime(csv[date_column])
    
#     filtered_data = csv[(csv[date_column] >= date_range[0])
#                         & (csv[date_column] <= date_range[1])]
#     filtered_data = filtered_data.reset_index(drop=True)
#     print(f"Filtered data length: {len(filtered_data[value_column])}")
#     return filtered_data

# def run_algorithm(csv_file_path, date_range, date_column, value_column, weight):
#     dataset = load_and_filter_data(
#         csv_file_path, date_range, date_column, value_column)
    
#     tfs = build_time_frames(dataset, date_column, value_column)
#     original_timeline_length = len(tfs)
    
#     print('Start processing MDL...')
#     start_time = time.time()
#     best_folded_timeline, min_dl = find_mdl(tfs, weight)
#     folded_timeline_length = len(best_folded_timeline)
#     elapsed_time = time.time() - start_time
#     print("Runtime: ", elapsed_time)
    
#     folded_dates = [point.start_point.time_value for point in best_folded_timeline]
#     folded_dates.append(best_folded_timeline[-1].end_point.time_value)
#     data = dataset[dataset[date_column].isin(folded_dates)]
    
#     return dataset, data, date_column, value_column, elapsed_time, original_timeline_length, folded_timeline_length

# def parse_arguments():
#     parser = argparse.ArgumentParser(description='Load and process time series data.')
#     parser.add_argument('csv_file_path', type=str, help='Path to the CSV file')
#     parser.add_argument('date_range', type=str, nargs=2, help='Start and end date in YYYY-MM-DD format')
#     parser.add_argument('date_column', type=str, help='Name of the date column in the dataset')
#     parser.add_argument('value_column', type=str, nargs='+', help='List of column names to be processed')
#     parser.add_argument('weight', type=float, help='Weight parameter for the MDL process')
#     return parser.parse_args()

# if __name__ == "__main__":
#     args = parse_arguments()
#     dataset, data, date_column, value_column, runtime, original_length, folded_length = run_algorithm(
#         args.csv_file_path, args.date_range, args.date_column, args.value_column, args.weight)
    
#     dataset.to_pickle('dataset.pkl')
#     data.to_pickle('data.pkl')
#     with open('meta.txt', 'w') as f:
#         f.write(f"{date_column}," + ",".join(value_column))
    
#     print(f"Algorithm runtime: {runtime} seconds")
#     print(f"Original timeline length: {original_length}")
#     print(f"Folded timeline length: {folded_length}")


import os
import time
import pandas as pd
from cluster_ts import find_mdl
from data import build_time_frames

def load_and_filter_data(csv_file_path, date_range, date_column, value_column):
    csv = pd.read_csv(csv_file_path)
    csv.dropna(inplace=True)
    csv[date_column] = pd.to_datetime(csv[date_column])
    
    filtered_data = csv[(csv[date_column] >= date_range[0])
                        & (csv[date_column] <= date_range[1])]
    filtered_data = filtered_data.reset_index(drop=True)
    print(f"Filtered data length: {len(filtered_data[value_column])}")
    return filtered_data

def run_algorithm(csv_file_path, date_range, date_column, value_column, weight):
    dataset = load_and_filter_data(
        csv_file_path, date_range, date_column, value_column)
    
    tfs = build_time_frames(dataset, date_column, value_column)
    original_timeline_length = len(tfs)
    
    print('Start processing MDL...')
    start_time = time.time()
    best_folded_timeline, min_dl = find_mdl(tfs, weight)
    folded_timeline_length = len(best_folded_timeline)
    elapsed_time = time.time() - start_time
    print("Runtime: ", elapsed_time)
    
    folded_dates = [point.start_point.time_value for point in best_folded_timeline]
    folded_dates.append(best_folded_timeline[-1].end_point.time_value)
    data = dataset[dataset[date_column].isin(folded_dates)]
    
    return dataset, data, date_column, value_column, elapsed_time, original_timeline_length, folded_timeline_length
