import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load the dataset
file_path = '../../backend/data/StnData_2020-2023_dailytemp.csv'  # Update this to your file path
data = pd.read_csv(file_path)

# Extract the 'Max' column
max_temps = data['Max'].values

# Normalize the data
scaler = MinMaxScaler(feature_range=(-1, 1))
max_temps = scaler.fit_transform(max_temps.reshape(-1, 1)).flatten()

# Parameters
timesteps = 10  # Number of time steps in each input sequence
num_features = 1  # Number of features in the input data (Max temperature)
latent_dim = 50  # Increased dimensionality of the latent space
epochs = 100  # Increased number of epochs for training
batch_size = 32  # Batch size for training

# Prepare the time series data for training
def create_sequences(data, timesteps):
    sequences = []
    for i in range(len(data) - timesteps):
        sequences.append(data[i:i + timesteps])
    return np.array(sequences)

X_train = create_sequences(max_temps, timesteps)
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

# Build and compile the RAE model
rae_model = build_rae_model(timesteps, num_features, latent_dim)
rae_model.compile(optimizer='adam', loss='mean_squared_error')

# Callback to monitor training
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='rae_model_best.keras',
    save_best_only=True,
    monitor='val_loss',
    mode='min'
)

# Train the model
history = rae_model.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[checkpoint_callback])

# Plot training & validation loss values
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Load the best model
rae_model = tf.keras.models.load_model('rae_model_best.keras')

def plot_reconstruction(original, reconstructed, title='Reconstructed vs Original'):
    plt.figure(figsize=(14, 7))
    plt.plot(original, label='Original')
    plt.plot(reconstructed, label='Reconstructed')
    plt.title(title)
    plt.legend()
    plt.show()

# Verify model performance
example_input = X_train[:1]
reconstructed_output = rae_model.predict(example_input)
reconstructed_output_rescaled = scaler.inverse_transform(reconstructed_output.flatten().reshape(-1, 1)).flatten()
original_input_rescaled = scaler.inverse_transform(example_input.flatten().reshape(-1, 1)).flatten()

plot_reconstruction(original_input_rescaled, reconstructed_output_rescaled, title='RAE Model Performance')

# Save the trained model
rae_model.save('rae_model.h5')
