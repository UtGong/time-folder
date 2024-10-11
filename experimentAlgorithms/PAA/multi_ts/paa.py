import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from pyts.approximation import PiecewiseAggregateApproximation

def main():
    start_time = time.time()

    # Parameters (adjust these as needed)
    file_path = '../../../backend/data/StnData_2020-2023_dailytemp.csv'
    window_size = 63  # PAA window size
    y_min = -40  # Set desired y-axis minimum, or None to auto-compute
    y_max = 120  # Set desired y-axis maximum, or None to auto-compute

    # Load the dataset
    data = pd.read_csv(file_path)

    # Ensure all data is numeric and handle missing values
    data = data.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Assuming each column is a different time series, transpose the data
    time_series_data = data.values.T

    # Define PAA parameters
    paa = PiecewiseAggregateApproximation(window_size=window_size)

    # Perform PAA
    compressed_data = paa.transform(time_series_data)

    # Convert the compressed data back to a DataFrame for easier handling
    compressed_df = pd.DataFrame(compressed_data)

    print("compressed_data", compressed_data)

    print('Original data shape:', time_series_data.shape)
    print('Compressed data shape:', compressed_data.shape)

    run_time = time.time() - start_time
    print('Runtime:', run_time)

    # Visualize the original and compressed time series
    plt.figure(figsize=(12, 6))

    # Plot compressed time series
    for i in range(1, compressed_data.shape[0]):
        plt.plot(np.linspace(0, time_series_data.shape[1], compressed_data.shape[1]), 
                 compressed_data[i], label=f'Compressed Series {i+1}')

    # Set y-axis limits if provided
    if y_min is not None and y_max is not None:
        plt.ylim(y_min, y_max)

    # plt.xlabel('Time')
    # plt.ylabel('Value')
    # plt.title('PAA')
    plt.title('PAA', fontsize=14)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.tight_layout()
    # legend (named as max and min instead of the original title)
    plt.legend(labels=['Max', 'Min'])
    
    plt.savefig('PAA_multits.png')
    print("Plot saved as 'PAA_multits.png'")

if __name__ == "__main__":
    main()
