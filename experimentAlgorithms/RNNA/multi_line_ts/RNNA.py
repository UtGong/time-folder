import numpy as np
import pandas as pd
import time
from scipy.interpolate import interp1d
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def main():
    start_time = time.time()

    # Parameters
    file_path = '../../../backend/data/StnData_2020-2023_dailytemp.csv'  # Update this to your file path
    max_deviation = 0.9  # Adjusted max deviation for more lenient compression
    rae_len = 5  # Adjusted length for the RAE model
    y_min = -40  # Set desired y-axis minimum
    y_max = 120  # Set desired y-axis maximum

    # Load the trained RAE model
    rae_model = load_model('rae_model.h5')

    # Load the dataset
    data = pd.read_csv(file_path)

    # Extract the 'Min' and 'Max' columns
    min_temps = data['Min'].values
    max_temps = data['Max'].values

    # Stack min and max temperatures as features
    temps = np.column_stack((min_temps, max_temps))

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    temps_scaled = scaler.fit_transform(temps)

    def interpolate(series, target_length):
        """Interpolates or downsamples a series to a target length."""
        current_length = series.shape[0]
        if current_length == target_length:
            return series
        x = np.linspace(0, current_length - 1, current_length)
        f = interp1d(x, series, axis=0, kind='linear')
        x_new = np.linspace(0, current_length - 1, target_length)
        return f(x_new)

    def adaptive_pairwise_compression(S, rae_model, max_deviation, rae_len):
        """Adaptive Pairwise Compression Algorithm."""
        data_len = S.shape[0]
        st = 0
        compressed_segments = []

        while st < data_len:
            input_len = data_len - st
            len_stride = input_len // 2
            last_valid_len = 0

            while st + input_len <= data_len and len_stride > 1:
                xt = S[st:st + input_len]
                x_tilde = interpolate(xt, rae_len)
                x_tilde = x_tilde.reshape((1, rae_len, S.shape[1]))  # Reshape for model prediction
                x_hat = rae_model.predict(x_tilde)[0]
                max_dev = np.max(np.abs(xt - interpolate(x_hat, input_len)))

                if max_dev <= max_deviation:
                    last_valid_len = input_len
                    input_len += len_stride
                else:
                    input_len -= len_stride

                len_stride //= 2

            xt = S[st:st + last_valid_len]
            x_tilde = interpolate(xt, rae_len)
            x_tilde = x_tilde.reshape((1, rae_len, S.shape[1]))  # Reshape for model prediction
            x_hat = rae_model.predict(x_tilde)[0]
            compressed_segments.append((st, last_valid_len, x_hat))  # Store the starting point, length, and compressed data
            st += last_valid_len

        return compressed_segments

    def reconstruct_series(compressed_segments, data_len):
        """Reconstruct the series from compressed segments."""
        reconstructed_series = np.zeros((data_len, compressed_segments[0][2].shape[1]))
        for (st, segment_len, x_hat) in compressed_segments:
            x_hat_interpolated = interpolate(x_hat, segment_len)
            reconstructed_series[st:st + segment_len] = x_hat_interpolated
        return reconstructed_series

    # Perform adaptive pairwise compression
    compressed_segments = adaptive_pairwise_compression(temps_scaled, rae_model, max_deviation, rae_len)
    compressed_series = [item[2] for item in compressed_segments]  # Extract only the compressed data

    # Flatten the list of compressed data segments and scale back
    compressed_series_flattened = np.concatenate(compressed_series, axis=0)
    compressed_series_scaled = scaler.inverse_transform(compressed_series_flattened)

    # Reconstruct the full series from the compressed segments
    reconstructed_series_scaled = reconstruct_series(compressed_segments, temps_scaled.shape[0])
    reconstructed_series = scaler.inverse_transform(reconstructed_series_scaled)

    run_time = time.time() - start_time

    print("Run time:", run_time)
    print("Length of compressed series:", compressed_series_flattened.shape[0])
    print("Length of original series:", temps.shape[0])

    # Plot the original and reconstructed series
    plt.figure(figsize=(12, 6))

    # Plot Min temperature
    plt.plot(reconstructed_series[:, 0], label='Reconstructed Min Temperature')

    # Plot Max temperature
    plt.plot(reconstructed_series[:, 1], label='Reconstructed Max Temperature')

    # Set y-axis limits
    plt.ylim(y_min, y_max)

    plt.title('PAA', fontsize=14)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.tight_layout()
    plt.legend(labels=['Max', 'Min'])
    plt.savefig("RNNA_ts_min_max.png")
    print("Plot saved as 'RNNA_ts_min_max.png'")

if __name__ == "__main__":
    main()
