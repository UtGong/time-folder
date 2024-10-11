import os
import time
import pandas as pd
from cluster_ts import find_mdl
from data import build_time_frames
from ts.plot import draw_init_line_plot, compute_y_axis_parameters, draw_merged_line_plot_non_linear

# Function to load and filter data
def load_and_filter_data(csv_file_path, date_range, date_column, value_column):
    csv = pd.read_csv(csv_file_path)
    csv.dropna(inplace=True)
    csv[date_column] = pd.to_datetime(csv[date_column])
    
    filtered_data = csv[(csv[date_column] >= date_range[0])
                        & (csv[date_column] <= date_range[1])]
    filtered_data = filtered_data.reset_index(drop=True)
    print(f"Filtered data length: {len(filtered_data[value_column])}")
    return filtered_data

# Function to run the algorithm
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

# Function to visualize the results
def visualize_results(y_min=None, y_max=None):
    try:
        if not os.path.exists('dataset.pkl'):
            print("dataset.pkl does not exist")
            return None, None
        
        if not os.path.exists('data.pkl'):
            print("data.pkl does not exist")
            return None, None

        # Load the data
        dataset = pd.read_pickle('dataset.pkl')
        data = pd.read_pickle('data.pkl')
        
        print("Data loaded successfully")
        # print("Dataset head:\n", dataset.head())
        print("Filtered data head:\n", data.head())
        
        # Load metadata
        with open('meta.txt', 'r') as f:
            meta = f.read().split(',')
        date_column = meta[0]
        value_column = [col.strip() for col in meta[1:]]
        
        # print(f"Date column: {date_column}")
        # print(f"Value columns: {value_column}")
        
        ts_type = 'multi' if len(value_column) > 1 else 'single'
        output_path = os.path.join('output/ts/', ts_type)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        # Compute or use provided y-axis limits
        global_min, global_max = dataset[value_column].min().min(), dataset[value_column].max().max()
        y_min = y_min if y_min is not None else global_min
        y_max = y_max if y_max is not None else global_max
        y_params = compute_y_axis_parameters(y_min, y_max)
        
        # Plot and save the initial data visualization
        print("Generating initial plot...")
        init_plt = draw_init_line_plot(dataset, date_column, value_column, *y_params)
        init_graph_path = os.path.join(output_path, "init.png")
        init_plt.savefig(init_graph_path)
        print(f"Initial graph saved at: {init_graph_path}")
        
        # Plot merged data and save it
        print("Generating folded plot...")
        plt = draw_init_line_plot(data, date_column, value_column, *y_params)
        folded_graph_path = os.path.join(output_path, "folded.png")
        plt.savefig(folded_graph_path)
        print(f"Folded graph saved at: {folded_graph_path}")
        
        print("Generating nonlinear folded plot...")
        nonlinear_plt = draw_merged_line_plot_non_linear(data, date_column, value_column, *y_params)
        nonlinear_graph_path = os.path.join(output_path, "folded_nonlinear.png")
        nonlinear_plt.savefig(nonlinear_graph_path)
        print(f"Nonlinear folded graph saved at: {nonlinear_graph_path}")
        
        return init_graph_path, folded_graph_path, nonlinear_graph_path
        
        # return init_graph_path, folded_graph_path

    except Exception as e:
        print("An error occurred while generating the visualization:", str(e))
        return None, None

def main():
    # Parameters (these can be controlled as needed)
    parameters = {
        "csv_file_path": "data/StnData_2020-2023_dailytemp.csv",
        "date_range": ["2020-01-01", "2023-12-31"],
        "date_column": "Date",
        "value_column": ["Max", "Min"],
        "weight": 3,
        "y_min": None,  # Set desired y-axis minimum, or None to compute dynamically
        "y_max": None   # Set desired y-axis maximum, or None to compute dynamically
    }
    
    csv_file_path = parameters["csv_file_path"]
    date_range = parameters["date_range"]
    date_column = parameters["date_column"]
    value_column = parameters["value_column"]
    weight = parameters["weight"]
    y_min = parameters["y_min"]
    y_max = parameters["y_max"]
    
    # Run ts_main function
    print("Running the algorithm...")
    dataset, data, date_column, value_column, runtime, original_length, folded_length = run_algorithm(
        csv_file_path, date_range, date_column, value_column, weight
    )
    
    # Save dataset and filtered data for visualization
    dataset.to_pickle('dataset.pkl')
    data.to_pickle('data.pkl')
    with open('meta.txt', 'w') as f:
        f.write(f"{date_column}," + ",".join(value_column))
    
    print(f"Algorithm runtime: {runtime} seconds")
    print(f"Original timeline length: {original_length}")
    print(f"Folded timeline length: {folded_length}")

    # Run visualization function with y-axis limits
    print("Visualizing the results...")
    # init_graph_path, folded_graph_path = visualize_results(y_min=y_min, y_max=y_max)
    
    # print(f"Initial graph saved at: {init_graph_path}")
    # print(f"Folded graph saved at: {folded_graph_path}")
    init_graph_path, folded_graph_path, nonlinear_graph_path = visualize_results(y_min=y_min, y_max=y_max)
    
    print(f"Initial graph saved at: {init_graph_path}")
    print(f"Folded graph saved at: {folded_graph_path}")
    print(f"Nonlinear folded graph saved at: {nonlinear_graph_path}")


if __name__ == "__main__":
    main()
