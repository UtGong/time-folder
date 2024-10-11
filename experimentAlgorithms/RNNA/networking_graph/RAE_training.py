import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load the dataset
file_path = '../../../backend/data/analysis_df.csv'  # Update this to your file path
data = pd.read_csv(file_path)

# Extract the 'Min' and 'Max' columns
min_temps = data['dep_count'].values
max_temps = data['arr_count'].values

# Stack min and max temperatures as features
temps = np.column_stack((min_temps, max_temps))

# Normalize the data
scaler = MinMaxScaler(feature_range=(-1, 1))
temps_scaled = scaler.fit_transform(temps)

# Parameters
timesteps = 20  # Number of time steps in each input sequence
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

# Save the trained model
rae_model.save('rae_model.h5')
