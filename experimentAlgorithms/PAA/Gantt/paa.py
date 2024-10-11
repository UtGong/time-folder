import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from pyts.approximation import PiecewiseAggregateApproximation

# Load the dataset
file_path = '../../../backend/data/HR.csv'
data = pd.read_csv(file_path)

# Function to categorize arrival times
on_time_threshold = '10:00:00'
late_threshold = '12:00:00'

def categorize_time(time_str):
    if pd.isnull(time_str) or time_str == '':
        return 'absence'
    time = pd.to_datetime(time_str).time()
    if time <= pd.to_datetime(on_time_threshold).time():
        return 'on-time'
    elif time <= pd.to_datetime(late_threshold).time():
        return 'late'
    else:
        return 'absence'

# Apply the function to categorize the times
for col in data.columns[1:]:
    data[col] = data[col].apply(categorize_time)

# Assign a numeric index to each status
status_mapping = {'on-time': 2, 'late': 1, 'absence': 0}
for col in data.columns[1:]:
    data[col] = data[col].map(status_mapping)

# Ensure all data is numeric and handle missing values
data = data.apply(pd.to_numeric, errors='coerce').fillna(0)

# Transform the data into time series format (each worker as a separate time series)
time_series_data = data.values[:, 1:]  # Exclude the first column (worker indices)

# Start time for measuring runtime
start_time = time.time()

# Define PAA parameters
window_size = 7  # Adjust this value to control the compression ratio
paa = PiecewiseAggregateApproximation(window_size=window_size)

# Perform PAA
compressed_data = paa.transform(time_series_data)

# Recategorize the PAA results
def recategorize(paa_result):
    recategorized = np.zeros_like(paa_result, dtype=int)
    for i in range(paa_result.shape[0]):
        for j in range(paa_result.shape[1]):
            value = paa_result[i, j]
            if value >= 1.5:
                recategorized[i, j] = 2  # on-time
            elif value >= 0.5:
                recategorized[i, j] = 1  # late
            else:
                recategorized[i, j] = 0  # absence
    return recategorized

recategorized_data = recategorize(compressed_data)

# Get the compressed dates
num_dates = data.shape[1] - 1  # Subtract 1 to exclude the worker index column
compressed_dates_indices = np.linspace(0, num_dates - 1, recategorized_data.shape[1], endpoint=True, dtype=int)
compressed_dates = data.columns[1:][compressed_dates_indices]

print('Original data shape:', time_series_data.shape)
print('Compressed data shape:', compressed_data.shape)

# Measure and print runtime
run_time = time.time() - start_time
print('Runtime:', run_time)

# Prepare data_long for plotting
data_long = data.melt(id_vars=[data.columns[0]], var_name='Date', value_name='Status Code')
data_long['Date'] = pd.to_datetime(data_long['Date'])
data_long['User Index'] = data_long[data.columns[0]]  # Add the user index column
data_long['Status'] = data_long['Status Code'].map({2: 'on-time', 1: 'late', 0: 'absence'})

# Function to plot worker status over time
def plot_worker_status(data_long, compressed_dates):
    plt.figure(figsize=(12, 6))
    
    # Ensure the x-axis contains the entire range of dates from the original dataset
    dates = pd.to_datetime(data_long['Date'].unique())
    plt.xticks(dates, rotation=45)
    
    # Plot only the compressed dates but keep the full x-axis timeline
    for date in compressed_dates:
        subset = data_long[data_long['Date'] == date]
        for idx, row in subset.iterrows():
            if row['Status'] == 'on-time':
                color = 'green'
            elif row['Status'] == 'late':
                color = 'red'
            else:
                continue  # Skip plotting for 'absence'
            plt.scatter(row['Date'], row['User Index'], color=color)

    plt.xlabel('Date')
    plt.ylabel('User Index')
    plt.title('Worker Status Over Time')
    plt.savefig('PAA_gantt.png')

plot_worker_status(data_long, compressed_dates)
