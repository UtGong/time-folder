import os
import time
import pandas as pd
import argparse
from cluster_ts import find_mdl
from data import build_time_frames
from ts.plot import draw_init_line_plot, draw_merged_line_plot_non_linear, draw_merged_line_plot, compute_y_axis_parameters

def load_and_filter_data(csv_file_path, date_range, date_column, value_column):
    # Load CSV and convert the specified date column to datetime
    csv = pd.read_csv(csv_file_path)
    csv.dropna(inplace=True)
    csv[date_column] = pd.to_datetime(csv[date_column])
    
    # Filter the data within the specified date range
    filtered_data = csv[(csv[date_column] >= date_range[0])
                        & (csv[date_column] <= date_range[1])]
    filtered_data = filtered_data.reset_index(drop=True)
    print(f"Filtered data length: {len(filtered_data[value_column])}")
    return filtered_data

def main(csv_file_path, date_range, date_column, value_column, weight):
    # Load and filter data based on user-specified parameters
    dataset = load_and_filter_data(
        csv_file_path, date_range, date_column, value_column)
    
    # Assuming build_time_frames and find_mdl are defined elsewhere
    tfs = build_time_frames(dataset, date_column, value_column)
    print('Start processing MDL...')
    start_time = time.time()
    best_folded_timeline, min_dl = find_mdl(tfs, weight)
    print("Length of best folded timeline:", len(best_folded_timeline))
    
    elapsed_time = time.time() - start_time
    print("Runtime: ", elapsed_time)

    # Define output path based on the data type and range
    ts_type = 'multi' if len(value_column) > 1 else 'single'
    folder_name = date_range[0] + "-" + date_range[1]
    output_path = os.path.join('output/ts/', ts_type, os.path.basename(csv_file_path).split('.')[0], folder_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Compute global minimum and maximum values across specified columns for plotting
    global_min, global_max = dataset[value_column].min().min(), dataset[value_column].max().max()
    print(global_min, global_max)
    y_params = compute_y_axis_parameters(global_min, global_max)
    
    # Plot and save the initial data visualization
    init_plt = draw_init_line_plot(dataset, date_column, value_column, *y_params)
    init_graph_path = os.path.join(output_path, "init.png")
    init_plt.savefig(init_graph_path)
    
    # Filter dataset based on the timeline from MDL analysis
    folded_dates = [point.start_point.time_value for point in best_folded_timeline]
    folded_dates.append(best_folded_timeline[-1].end_point.time_value)
    data = dataset[dataset[date_column].isin(folded_dates)]
    
    # Plot merged data and save it
    plt = draw_init_line_plot(data, date_column, value_column, *y_params)
    folded_graph_path = os.path.join(output_path, "folded.png")
    plt.savefig(folded_graph_path)
    
    return init_graph_path, folded_graph_path
    # plt = draw_merged_line_plot_non_linear(data, date_column, value_column, *y_params)
    # plt.savefig(os.path.join(output_path, 'merged-nonlinear.png'))

def parse_arguments():
    # Setup command-line argument parsing
    parser = argparse.ArgumentParser(description='Load and process time series data.')
    parser.add_argument('csv_file_path', type=str, help='Path to the CSV file')
    parser.add_argument('date_range', type=str, nargs=2, help='Start and end date in YYYY-MM-DD format')
    parser.add_argument('date_column', type=str, help='Name of the date column in the dataset')
    parser.add_argument('value_column', type=str, nargs='+', help='List of column names to be processed')
    parser.add_argument('weight', type=float, help='Weight parameter for the MDL process')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    main(args.csv_file_path, args.date_range, args.date_column, args.value_column)