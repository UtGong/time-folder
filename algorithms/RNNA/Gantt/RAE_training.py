import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

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

# Prepare the sequences for training
timesteps = 10
def create_sequences(data, timesteps):
    sequences = []
    for i in range(len(data) - timesteps):
        sequences.append(data[i:i + timesteps])
    return np.array(sequences)

X_train = create_sequences(scaled_data, timesteps)
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
latent_dim = 50
rae_model = build_rae_model(timesteps, 1, latent_dim)  # Adjust num_features to 1
rae_model.compile(optimizer='adam', loss='mean_squared_error')

# Callback to monitor training
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='rae_model_best.keras',
    save_best_only=True,
    monitor='val_loss',
    mode='min'
)

# Train the model
history = rae_model.fit(X_train, X_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[checkpoint_callback])

# Plot training & validation loss values
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# # Load the best model
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
