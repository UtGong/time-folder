import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load the dataset
file_path = '../../../backend/data/StnData_2020-2023_dailytemp.csv'  # Update this to your file path
data = pd.read_csv(file_path)

# Extract the 'Min' and 'Max' columns
min_temps = data['Min'].values
max_temps = data['Max'].values

# Stack min and max temperatures as features
temps = np.column_stack((min_temps, max_temps))

# Normalize the data
scaler = MinMaxScaler(feature_range=(-1, 1))
temps_scaled = scaler.fit_transform(temps)

# Parameters
timesteps = 5  # Number of time steps in each input sequence
num_features = 2  # Number of features in the input data (Min and Max temperature)
latent_dim = 50  # Increased dimensionality of the latent space
epochs = 200  # Increased number of epochs for training
batch_size = 64  # Batch size for training

# Prepare the time series data for training
def create_sequences(data, timesteps):
    sequences = []
    for i in range(len(data) - timesteps):
        sequences.append(data[i:i + timesteps])
    return np.array(sequences)

X_train = create_sequences(temps_scaled, timesteps)

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

def plot_reconstruction(original, reconstructed, feature_index, feature_name):
    plt.figure(figsize=(14, 7))
    plt.plot(original[:, feature_index], label=f'Original {feature_name}')
    plt.plot(reconstructed[:, feature_index], label=f'Reconstructed {feature_name}')
    plt.title(f'Reconstructed vs Original {feature_name}')
    plt.legend()
    plt.show()

# Verify model performance
example_input = X_train[:1]
reconstructed_output = rae_model.predict(example_input)[0]
reconstructed_output_rescaled = scaler.inverse_transform(reconstructed_output)
original_input_rescaled = scaler.inverse_transform(example_input[0])

# Plot reconstruction for Min and Max temperatures
plot_reconstruction(original_input_rescaled, reconstructed_output_rescaled, feature_index=0, feature_name='Min Temperature')
plot_reconstruction(original_input_rescaled, reconstructed_output_rescaled, feature_index=1, feature_name='Max Temperature')

# Save the trained model
rae_model.save('rae_model.h5')
