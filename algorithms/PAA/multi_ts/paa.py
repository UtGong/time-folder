import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from pyts.approximation import PiecewiseAggregateApproximation

start_time = time.time()

# Load the dataset
file_path = '../../../backend/data/StnData_2020-2023_dailytemp.csv'
data = pd.read_csv(file_path)

# Ensure all data is numeric and handle missing values
data = data.apply(pd.to_numeric, errors='coerce').fillna(0)

# Assuming each column is a different time series, transpose the data
time_series_data = data.values.T

# Define PAA parameters
window_size = 63
paa = PiecewiseAggregateApproximation(window_size=window_size)

# Perform PAA
compressed_data = paa.transform(time_series_data)

# Convert the compressed data back to a DataFrame for easier handling
compressed_df = pd.DataFrame(compressed_data)

print('Original data shape:', time_series_data.shape)
print('Compressed data shape:', compressed_data.shape)

run_time = time.time() - start_time

print('Runtime:', run_time)

# Visualize the original and compressed time series
plt.figure(figsize=(14, 7))

# Plot compressed time series
for i in range(1, compressed_data.shape[0]):
    plt.plot(np.linspace(0, time_series_data.shape[1], compressed_data.shape[1]), 
             compressed_data[i], label=f'Compressed Series {i+1}')

plt.xlabel('Time')
plt.ylabel('Value')
plt.title('PAA')
# plt.legend()
plt.savefig('PAA_multits.png')

