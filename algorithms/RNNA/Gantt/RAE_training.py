import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

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

# Assign a numeric index to each status
status_mapping = {'on-time': 0, 'late': 1, 'absence': 2}
data_long['Status Code'] = data_long['Status'].map(status_mapping)

# Extract the status code column
S = data_long['Status Code'].values

# Normalize the data
scaler = MinMaxScaler(feature_range=(-1, 1))
S_scaled = scaler.fit_transform(S.reshape(-1, 1)).flatten()

# Parameters
timesteps = 10  # Number of time steps in each input sequence
num_features = 1  # Number of features in the input data (Status Code)
latent_dim = 50  # Dimensionality of the latent space
epochs = 100  # Number of epochs for training
batch_size = 32  # Batch size for training

# Prepare the time series data for training
def create_sequences(data, timesteps):
    sequences = []
    for i in range(len(data) - timesteps):
        sequences.append(data[i:i + timesteps])
    return np.array(sequences)

X_train = create_sequences(S_scaled, timesteps)
X_train = np.expand_dims(X_train, axis=-1)  # Add feature dimension

# Define the RAE model
def build_rae_model(timesteps, num_features, latent_dim):
    inputs = Input(shape=(timesteps, num_features))
    encoded = LSTM(latent_dim, activation='relu', return_sequences=True)(inputs)
    encoded = LSTM(latent_dim, activation='relu')(encoded)
    latent = RepeatVector(timesteps)(encoded)
    decoded = LSTM(latent_dim, activation='relu', return_sequences=True)(latent)
    decoded = LSTM(latent_dim, activation='relu', return_sequences=True)(decoded)
    outputs = TimeDistributed(Dense(num_features))(decoded)
    model = Model(inputs, outputs)
    return model

rae_model = build_rae_model(timesteps, num_features, latent_dim)
rae_model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = rae_model.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)

# Save the trained model
rae_model.save('rae_model.h5')

# Plot training & validation loss values
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
