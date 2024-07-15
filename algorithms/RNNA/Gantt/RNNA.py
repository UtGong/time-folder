import pandas as pd
import numpy as np
import time
from scipy.interpolate import interp1d
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

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

# Transform the data to long format for easier processing
data_long = data.melt(id_vars=['Unnamed: 0'], var_name='Date', value_name='Status')
data_long.rename(columns={'Unnamed: 0': 'User Index'}, inplace=True)

# Convert the 'Date' column to datetime format
data_long['Date'] = pd.to_datetime(data_long['Date'])

# Assign a numeric index to each status
status_mapping = {'on-time': 0, 'late': 1, 'absence': 2}
data_long['Status Code'] = data_long['Status'].map(status_mapping)

# Adjust User Index to start from 0
data_long['User Index'] = data_long['User Index'] - 1

# algorithm started, marked current time as start time
start_time = time.time()

# Extract the status code column
S = data_long['Status Code'].values

# Normalize the data
scaler = MinMaxScaler(feature_range=(-1, 1))
S_scaled = scaler.fit_transform(S.reshape(-1, 1)).flatten()

# Load the trained RAE model
rae_model = load_model('rae_model.h5')

def interpolate(series, target_length):
    """Interpolates or downsamples a series to a target length."""
    current_length = len(series)
    if current_length == target_length:
        return series
    x = np.linspace(0, current_length - 1, current_length)
    f = interp1d(x, series, kind='linear')
    x_new = np.linspace(0, current_length - 1, target_length)
    return f(x_new)

def adaptive_pairwise_compression(S, rae_model, max_deviation, rae_len):
    """Adaptive Pairwise Compression Algorithm."""
    data_len = len(S)
    st = 0
    compressed_segments = []

    while st < data_len:
        input_len = data_len - st
        len_stride = input_len // 2
        last_valid_len = 0

        while st + input_len <= data_len and len_stride > 1:
            xt = S[st:st + input_len]
            if len(xt) == 0:
                break  # Skip empty segments
            x_tilde = interpolate(xt, rae_len)
            x_tilde = x_tilde.reshape((1, rae_len, 1))  # Reshape for model prediction
            x_hat = rae_model.predict(x_tilde)[0].flatten()
            max_dev = np.max(np.abs(xt - interpolate(x_hat, input_len)))

            if max_dev <= max_deviation:
                last_valid_len = input_len
                input_len += len_stride
            else:
                input_len -= len_stride

            len_stride //= 2

        xt = S[st:st + last_valid_len]
        if len(xt) == 0:
            break  # Skip empty segments
        x_tilde = interpolate(xt, rae_len)
        x_tilde = x_tilde.reshape((1, rae_len, 1))  # Reshape for model prediction
        x_hat = rae_model.predict(x_tilde)[0].flatten()
        compressed_segments.append((st, last_valid_len, x_hat))  # Store the starting point, length, and compressed data
        st += last_valid_len

    return compressed_segments

# Adjust the parameters for compression
max_deviation = 1.7  
rae_len = 10  

compressed_segments = adaptive_pairwise_compression(S_scaled, rae_model, max_deviation, rae_len)
compressed_series = [item[2] for item in compressed_segments]

# Flatten the list of compressed data segments and scale back
compressed_series_flattened = np.concatenate(compressed_series)
compressed_series_scaled = scaler.inverse_transform(compressed_series_flattened.reshape(-1, 1)).flatten()

compressed_dates = data_long.loc[[seg[0] for seg in compressed_segments if len(seg[2]) > 0], 'Date'].values

run_time = time.time() - start_time
print("Run time: ", run_time)
print("length of original data: ", len(S_scaled))
print("length of compressed_dates: ", len(compressed_dates))

def plot_worker_status(data_long, compressed_dates):
    plt.figure(figsize=(14, 7))
    
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
    plt.savefig('RNNA_gantt.png')

plot_worker_status(data_long, compressed_dates)
