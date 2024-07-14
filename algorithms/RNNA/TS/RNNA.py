import numpy as np
from scipy.interpolate import interp1d
from tensorflow.keras.models import load_model
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the trained RAE model
rae_model = load_model('rae_model.h5')

# Load the dataset
file_path = '../../backend/data/StnData_2020-2023_dailytemp.csv'  # Update this to your file path
data = pd.read_csv(file_path)

# Extract the 'Max' column
S = data['Max'].values

# Normalize the data
scaler = MinMaxScaler(feature_range=(-1, 1))
S_scaled = scaler.fit_transform(S.reshape(-1, 1)).flatten()

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
        x_tilde = interpolate(xt, rae_len)
        x_tilde = x_tilde.reshape((1, rae_len, 1))  # Reshape for model prediction
        x_hat = rae_model.predict(x_tilde)[0].flatten()
        compressed_segments.append((st, last_valid_len, x_hat))  # Store the starting point, length, and compressed data
        st += last_valid_len

    return compressed_segments

def reconstruct_series(compressed_segments, data_len):
    """Reconstruct the series from compressed segments."""
    reconstructed_series = np.zeros(data_len)
    for (st, segment_len, x_hat) in compressed_segments:
        x_hat_interpolated = interpolate(x_hat, segment_len)
        reconstructed_series[st:st + segment_len] = x_hat_interpolated
    return reconstructed_series

# Set the parameters for compression
max_deviation = 0.9  # Adjusted max deviation for more lenient compression
rae_len = 10

compressed_segments = adaptive_pairwise_compression(S_scaled, rae_model, max_deviation, rae_len)
compressed_series = [item[2] for item in compressed_segments]  # Extract only the compressed data

# Flatten the list of compressed data segments and scale back
compressed_series_flattened = np.concatenate(compressed_series)
compressed_series_scaled = scaler.inverse_transform(compressed_series_flattened.reshape(-1, 1)).flatten()

# Reconstruct the full series from the compressed segments
reconstructed_series_scaled = reconstruct_series(compressed_segments, len(S_scaled))
reconstructed_series = scaler.inverse_transform(reconstructed_series_scaled.reshape(-1, 1)).flatten()

print("Original series:", S)
print("Compressed series:", compressed_series_scaled)
print("Length of compressed series:", len(compressed_series_flattened))
print("Length of original series:", len(S))
print("Reconstructed series:", reconstructed_series)

# Plot the original and reconstructed series
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 7))
# plt.plot(S, label='Original Series')
plt.plot(reconstructed_series, label='Reconstructed Series')
plt.title('Original vs Reconstructed Series')
plt.legend()
plt.savefig("RNNA_ts.png")
