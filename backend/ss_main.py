import os
import time
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import argparse
from data import build_time_frames
from cluster import find_mdl
from symbolic_seq.data_transformation import transform_HR_data

def main(data_name, weight):
    # Starting the data transformation process
    print("Starting data transformation...")
    start_time = time.time()
    df, unique_id = transform_HR_data()  # Transform HR data (assumed to be a placeholder function)
    elapsed_time = time.time() - start_time
    print("Done transforming data...")
    print("Runtime: ", elapsed_time)

    # Create the output directory if it does not exist
    output_path = os.path.join('output/symbolic_seq/', data_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Plotting initial data status and saving the figure
    fig = plotting_with_status(df, 'time_value', 'user_task_status')
    init_graph_path = os.path.join(output_path, "init.png")
    fig.savefig(init_graph_path)
    
    # Prepare the data for minimum description length (MDL) process
    data_file_name = 'ss_to_ts'
    csv_file_path = os.path.join('data', data_file_name + '.csv')
    date_column = 'time_value'
    value_column = ["tp_value"]
    
    dataset = pd.read_csv(csv_file_path)
    print("len(original)", len(dataset))
    
    start_time = time.time()
    print("Start find mdl process...")
    tfs = build_time_frames(dataset, date_column, value_column)
    best_folded_timeline, min_dl = find_mdl(tfs, weight=weight)
    print("Done finding mdl process...")
    elapsed_time = time.time() - start_time
    print("len(best_folded_timeline)", len(best_folded_timeline))
    print("Runtime: ", elapsed_time)
    
    # Filter and plot the data based on best_folded_timeline
    dates = [point.start_point.time_value for point in best_folded_timeline]
    dates.append(best_folded_timeline[-1].end_point.time_value)
    
    df = df[df['time_value'].isin(dates)]
    fig = plotting_with_status(df, 'time_value', 'user_task_status')
    folded_graph_path = os.path.join(output_path, "folded.png")
    fig.savefig(folded_graph_path)
    
    return init_graph_path, folded_graph_path

def plotting_with_status(df, time_column, user_task_status_column):
    # Initialize the plot settings
    color_map = {'on time': 'green', 'late': 'red'}
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Loop through each data row for plotting
    for index, row in df.iterrows():
        date = pd.to_datetime(row[time_column])
        statuses = row[user_task_status_column]
        for user_index, status in enumerate(statuses):
            if status != 'Absence':
                ax.scatter(date, user_index, color=color_map.get(status, 'blue'), alpha=0.6)

    # Setting the x-axis with date formatting
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.set_ylim(-1, 15)
    ax.set_xlabel("Time Value")
    ax.set_ylabel("User Index")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend().set_visible(False)
    return plt

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('data_name', type=str, help='Name of the data set')
    parser.add_argument('weight', type=float, help='Weight parameter for the MDL process')
    return parser.parse_args()
    
if __name__ == "__main__":
    args = parse_arguments()
    main(args.data_name, args.weight)