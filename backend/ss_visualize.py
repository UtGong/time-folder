import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plotting_with_status(df, time_column, user_task_status_column):
    color_map = {'on time': 'green', 'late': 'red'}
    fig, ax = plt.subplots(figsize=(12, 6))

    for index, row in df.iterrows():
        date = pd.to_datetime(row[time_column])
        statuses = row[user_task_status_column]
        for user_index, status in enumerate(statuses):
            if status != 'Absence':
                ax.scatter(date, user_index, color=color_map.get(status, 'blue'), alpha=0.6)

    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.set_ylim(-1, 15)
    ax.set_xlabel("Time Value")
    ax.set_ylabel("User Index")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend().set_visible(False)
    return plt

def visualize_results():
    df = pd.read_pickle('transformed_data.pkl')
    filtered_df = pd.read_pickle('filtered_data.pkl')

    with open('meta_ss.txt', 'r') as f:
        meta = f.read().split(',')
    data_name = meta[0]
    weight = float(meta[1])
    output_path = meta[2]

    fig = plotting_with_status(df, 'time_value', 'user_task_status')
    init_graph_path = os.path.join(output_path, "init.png")
    fig.savefig(init_graph_path)

    fig = plotting_with_status(filtered_df, 'time_value', 'user_task_status')
    folded_graph_path = os.path.join(output_path, "folded.png")
    fig.savefig(folded_graph_path)

    return init_graph_path, folded_graph_path

if __name__ == "__main__":
    init_graph_path, folded_graph_path = visualize_results()
    print(f"Initial graph saved at: {init_graph_path}")
    print(f"Folded graph saved at: {folded_graph_path}")
