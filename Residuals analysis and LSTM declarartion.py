# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 12:13:00 2024

@author: pahar
"""



lb_test = acorr_ljungbox(residual, lags=[10], return_df=True)
print(lb_test)

from scipy.stats import shapiro
stat, p_value = shapiro(residual)
print('p-value:', p_value)

from scipy.signal import periodogram
f, Pxx_den = periodogram(residual)
plt.semilogy(f, Pxx_den)
plt.xlabel('Frequency')
plt.ylabel('Power')
plt.show()



def df_to_X_y(df, window_size=5):
  df_as_np = df.to_numpy()
  X = []
  y = []
  for i in range(len(df_as_np)-window_size):
    row = [[a] for a in df_as_np[i:i+window_size]]
    X.append(row)
    label = df_as_np[i+window_size]
    y.append(label)
  return np.array(X), np.array(y)

WINDOW_SIZE = 4
X1, y1 = df_to_X_y(residual, WINDOW_SIZE)
X1.shape, y1.shape

X_train1, y_train1 = X1[:350], y1[:350]
X_val1, y_val1 = X1[350:370], y1[350:370]
X_test1, y_test1 = X1[370:], y1[370:]
X_train1.shape, y_train1.shape, X_val1.shape, y_val1.shape, X_test1.shape, y_test1.shape

X_train1.ndim

timesteps = 4  # number of timesteps in each sequence
features = 1   # number of features per timestep

# Define model architecture
input_layer = Input(shape=(timesteps, features))  # Input shape should match your data
lstm_output = LSTM(3)(input_layer)                # Output 3-dimensional LSTM output
dense_1 = Dense(8, activation='relu')(lstm_output) # Dense layer with 8 units and ReLU activation
output_layer = Dense(1, activation='linear')(dense_1)  # Final output layer for regression

# Create the model
model_LSTM = Model(inputs=input_layer, outputs=output_layer)

# Print model summary
model_LSTM.summary()

model_LSTM.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])

def create_base_lstm_model(timesteps, features, lstm_units, dense_units, learning_rate=0.001):
    """Create the base LSTM model with configurable parameters"""
    input_layer = Input(shape=(timesteps, features))
    lstm_output = LSTM(int(lstm_units))(input_layer)
    dense_1 = Dense(dense_units, activation='relu')(lstm_output)
    output = Dense(1, activation='linear')(dense_1)

    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')

    return model

def train_evaluate_model(model, X_train, y_train, X_val, y_val, batch_size=32, epochs=100):
    """Train and evaluate the model"""
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[early_stopping],
        verbose=0
    )

    return model, history.history['val_loss'][-1]