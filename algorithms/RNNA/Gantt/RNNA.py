# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# import tensorflow as tf
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
# from scipy.interpolate import interp1d
# import matplotlib.pyplot as plt

# # Load the dataset
# file_path = '../../../backend/data/HR.csv'  # Update this to your file path
# hr_data = pd.read_csv(file_path)

# # Define the function to categorize each cell
# def categorize_time(time):
#     if pd.isnull(time):
#         return 'Absence'
#     time_part = pd.to_datetime(time).time()
#     if time_part >= pd.to_datetime('10:00:00').time():
#         return 'late'
#     else:
#         return 'on time'

# # Apply the categorization function to the dataset
# categorized_data = hr_data.applymap(categorize_time)

# # Encode the categories into numerical values
# category_mapping = {'Absence': 0, 'on time': 1, 'late': 2}
# numerical_data = categorized_data.replace(category_mapping)

# # Flatten the data and scale it
# flattened_data = numerical_data.values.flatten()
# scaler = MinMaxScaler(feature_range=(-1, 1))
# scaled_data = scaler.fit_transform(flattened_data.reshape(-1, 1)).flatten()

# # Load the best model
# rae_model = tf.keras.models.load_model('rae_model_best.keras')

# # Function to interpolate or downsample a series to a target length
# def interpolate(series, target_length):
#     current_length = len(series)
#     if current_length == target_length:
#         return series
#     x = np.linspace(0, current_length - 1, current_length)
#     f = interp1d(x, series, kind='linear')
#     x_new = np.linspace(0, current_length - 1, target_length)
#     return f(x_new)

# # Adaptive Pairwise Compression Algorithm
# def adaptive_pairwise_compression(S, rae_model, max_deviation, rae_len):
#     data_len = len(S)
#     st = 0
#     compressed_segments = []

#     while st < data_len:
#         input_len = data_len - st
#         len_stride = input_len // 2
#         last_valid_len = 0

#         while st + input_len <= data_len and len_stride > 1:
#             if input_len <= 1:  # Ensure input_len is greater than 1 to avoid empty segments
#                 break
#             xt = S[st:st + input_len]
#             x_tilde = interpolate(xt, rae_len)
#             x_tilde = x_tilde.reshape((1, rae_len, 1))  # Reshape for model prediction
#             x_hat = rae_model.predict(x_tilde)[0].flatten()
#             max_dev = np.max(np.abs(xt - interpolate(x_hat, input_len)))

#             if max_dev <= max_deviation:
#                 last_valid_len = input_len
#                 input_len += len_stride
#             else:
#                 input_len -= len_stride

#             len_stride //= 2

#         if last_valid_len == 0:  # If no valid length was found, break the loop to avoid infinite loop
#             break

#         xt = S[st:st + last_valid_len]
#         x_tilde = interpolate(xt, rae_len)
#         x_tilde = x_tilde.reshape((1, rae_len, 1))  # Reshape for model prediction
#         x_hat = rae_model.predict(x_tilde)[0].flatten()
#         compressed_segments.append((st, last_valid_len, x_hat))  # Store the starting point, length, and compressed data
#         st += last_valid_len

#     return compressed_segments

# # Function to reconstruct the series from compressed segments
# def reconstruct_series(compressed_segments, data_len):
#     reconstructed_series = np.zeros(data_len)
#     for (st, segment_len, x_hat) in compressed_segments:
#         x_hat_interpolated = interpolate(x_hat, segment_len)
#         reconstructed_series[st:st + segment_len] = x_hat_interpolated
#     return reconstructed_series

# # Set the parameters for compression
# max_deviation = 0.9
# rae_len = 10

# compressed_segments = adaptive_pairwise_compression(scaled_data, rae_model, max_deviation, rae_len)
# compressed_series = [item[2] for item in compressed_segments]

# # Flatten the list of compressed data segments and scale back
# compressed_series_flattened = np.concatenate(compressed_series)
# compressed_series_scaled = scaler.inverse_transform(compressed_series_flattened.reshape(-1, 1)).flatten()

# # Reconstruct the full series from the compressed segments
# reconstructed_series_scaled = reconstruct_series(compressed_segments, len(scaled_data))
# reconstructed_series = scaler.inverse_transform(reconstructed_series_scaled.reshape(-1, 1)).flatten()

# print("Original series:", scaled_data)
# print("Compressed series:", compressed_series_scaled)
# print("Length of compressed series:", len(compressed_series_flattened))
# print("Length of original series:", len(scaled_data))
# print("Reconstructed series:", reconstructed_series)

# # Plot the original and reconstructed series
# plt.figure(figsize=(14, 7))
# plt.plot(reconstructed_series, label='Reconstructed Series')
# plt.title('Original vs Reconstructed Series')
# plt.legend()
# plt.savefig("RNNA_gantt.png")




















































import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load the dataset
file_path = '../../../backend/data/HR.csv'  # Update this to your file path
hr_data = pd.read_csv(file_path)

# Define the function to categorize each cell
def categorize_time(time):
    if pd.isnull(time):
        return 'Absence'
    time_part = pd.to_datetime(time).time()
    if time_part >= pd.to_datetime('10:00:00').time():
        return 'late'
    else:
        return 'on time'

# Apply the categorization function to the dataset
categorized_data = hr_data.applymap(categorize_time)

# Encode the categories into numerical values
category_mapping = {'Absence': 0, 'on time': 1, 'late': 2}
numerical_data = categorized_data.replace(category_mapping)

# Flatten the data and scale it
flattened_data = numerical_data.values.flatten()
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_data = scaler.fit_transform(flattened_data.reshape(-1, 1)).flatten()

# Load the best model
rae_model = tf.keras.models.load_model('rae_model_best.keras')

# Function to interpolate or downsample a series to a target length
def interpolate(series, target_length):
    current_length = len(series)
    if current_length == target_length:
        return series
    x = np.linspace(0, current_length - 1, current_length)
    f = interp1d(x, series, kind='linear')
    x_new = np.linspace(0, current_length - 1, target_length)
    return f(x_new)

# Adaptive Pairwise Compression Algorithm
def adaptive_pairwise_compression(S, rae_model, max_deviation, rae_len):
    data_len = len(S)
    st = 0
    compressed_segments = []

    while st < data_len:
        input_len = data_len - st
        len_stride = input_len // 2
        last_valid_len = 0

        while st + input_len <= data_len and len_stride > 1:
            if input_len <= 1:  # Ensure input_len is greater than 1 to avoid empty segments
                break
            xt = S[st:st + input_len]
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

        if last_valid_len == 0:  # If no valid length was found, break the loop to avoid infinite loop
            break

        xt = S[st:st + last_valid_len]
        x_tilde = interpolate(xt, rae_len)
        x_tilde = x_tilde.reshape((1, rae_len, 1))  # Reshape for model prediction
        x_hat = rae_model.predict(x_tilde)[0].flatten()
        compressed_segments.append((st, last_valid_len, x_hat))  # Store the starting point, length, and compressed data
        st += last_valid_len

    return compressed_segments

# Function to reconstruct the series from compressed segments
def reconstruct_series(compressed_segments, data_len):
    reconstructed_series = np.zeros(data_len)
    for (st, segment_len, x_hat) in compressed_segments:
        x_hat_interpolated = interpolate(x_hat, segment_len)
        reconstructed_series[st:st + segment_len] = x_hat_interpolated
    return reconstructed_series

# Set the parameters for compression
max_deviation = 0.9
rae_len = 10

compressed_segments = adaptive_pairwise_compression(scaled_data, rae_model, max_deviation, rae_len)
compressed_series = [item[2] for item in compressed_segments]

# Flatten the list of compressed data segments and scale back
compressed_series_flattened = np.concatenate(compressed_series)
compressed_series_scaled = scaler.inverse_transform(compressed_series_flattened.reshape(-1, 1)).flatten()

# Reconstruct the full series from the compressed segments
reconstructed_series_scaled = reconstruct_series(compressed_segments, len(scaled_data))
reconstructed_series = scaler.inverse_transform(reconstructed_series_scaled.reshape(-1, 1)).flatten()

print("Original series:", scaled_data)
print("Compressed series:", compressed_series_scaled)
print("Length of compressed series:", len(compressed_series_flattened))
print("Length of original series:", len(scaled_data))
print("Reconstructed series:", reconstructed_series)

# Check lengths of data columns and flattened values
print(f"Number of time values: {len(hr_data.columns[1:])}")
print(f"Number of status values: {len(numerical_data.values.T.flatten())}")

# Create a DataFrame for the original data with time and status
time_values = pd.to_datetime(hr_data.columns[1:])  # Convert column names to datetime
status_values = numerical_data.values.T.flatten()

# Ensure time_values and status_values have the same length
min_length = min(len(time_values), len(status_values))
time_values = time_values[:min_length]
status_values = status_values[:min_length]

# Calculate the number of workers and the number of days
num_workers = numerical_data.shape[0]
num_days = numerical_data.shape[1]

# Ensure status_2d_array and time_values have matching lengths
if len(status_values) > num_days * num_workers:
    status_values = status_values[:num_days * num_workers]
elif len(status_values) < num_days * num_workers:
    status_values = np.pad(status_values, (0, num_days * num_workers - len(status_values)), constant_values=category_mapping['Absence'])

# Reshape the status_values to a 2D array where each row is a day and each column is a worker
status_2d_array = status_values.reshape(num_days, num_workers)

# Ensure time_values has the same length as num_days
if len(time_values) > num_days:
    time_values = time_values[:num_days]

# Create DataFrame for original data
original_data = pd.DataFrame(status_2d_array, columns=[f'Worker {i}' for i in range(num_workers)], index=time_values)

# Map numerical values back to their categories for plotting
status_mapping = {0: 'Absence', 1: 'on time', 2: 'late'}

# Function to plot data
def plot_status(df, title):
    color_map = {'on time': 'green', 'late': 'red', 'Absence': 'gray'}
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for worker in df.columns:
        for date, status in df[worker].items():  # Corrected from iteritems() to items()
            ax.scatter(date, int(worker.split()[1]), color=color_map[status_mapping[status]], alpha=0.6)

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.set_ylim(-1, len(df.columns))
    ax.set_xlabel("Time Value")
    ax.set_ylabel("Worker Index")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.title(title)
    plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', label=status, 
                                    markersize=10, markerfacecolor=color) for status, color in color_map.items()])
    plt.show()

# Plot the original data
plot_status(original_data, "Original Data")

# For the compressed data, create a similar DataFrame
compressed_status_values = compressed_series_scaled.reshape(-1, num_workers)
compressed_data = pd.DataFrame(compressed_status_values, columns=[f'Worker {i}' for i in range(num_workers)], index=time_values[:compressed_status_values.shape[0]])

# Plot the compressed data
plot_status(compressed_data, "Compressed Data")
