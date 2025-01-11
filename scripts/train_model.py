import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

def split_data(data, split_ratio=0.8):
    """
    Splits the dataset into training and test sets based on the split ratio.
    """
    train_size = int(len(data) * split_ratio)
    return data[:train_size], data[train_size:]

def scale_data(train_data, test_data, feature_columns, target_column):
    """
    Scales features and target using MinMaxScaler.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Scale features
    scaled_train = scaler.fit_transform(train_data[feature_columns])
    scaled_test = scaler.transform(test_data[feature_columns])
    
    # Scale target
    target_train = target_scaler.fit_transform(train_data[[target_column]])
    target_test = target_scaler.transform(test_data[[target_column]])
    
    return scaled_train, scaled_test, target_train, target_test, scaler, target_scaler

def create_sequences(data, target, timesteps=10):
    """
    Creates sequences of timesteps for the LSTM model.
    """
    X, y = [], []
    for i in range(len(data) - timesteps):
        X.append(data[i:i+timesteps])
        y.append(target[i+timesteps])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    """
    Defines the LSTM model architecture.
    """
    model = Sequential([
        LSTM(units=64, return_sequences=False, input_shape=input_shape),
        Dropout(0.3),
        Dense(units=32, activation='relu'),
        Dense(units=1)
    ])
    model.compile(optimizer='rmsprop', loss='mean_squared_error')
    return model

def evaluate_model(y_true, y_pred):
    """
    Evaluates the model using MAE, RMSE, and R².
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

if __name__ == "__main__":
    # Load preprocessed data
    data = pd.read_csv(r"C:\Users\USER\OneDrive\Dokumenter\data\processed/feature_engineered_data.csv")
    
    # Split data into train and test sets
    train_data, test_data = split_data(data)
    
    # Feature columns and target column
    feature_columns = ['Close*', 'Volume', 'EMA_10']
    target_column = 'Close*'
    
    # Scale data
    scaled_train, scaled_test, target_train, target_test, scaler, target_scaler = scale_data(
        train_data, test_data, feature_columns, target_column
    )
    
    # Create sequences
    timesteps = 10
    X_train, y_train = create_sequences(scaled_train, target_train, timesteps)
    X_test, y_test = create_sequences(scaled_test, target_test, timesteps)
    
    # Build LSTM model
    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    
    # Train the model
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)
    
    # Make predictions
    predicted_scaled = model.predict(X_test)
    predicted = target_scaler.inverse_transform(predicted_scaled)
    y_test_true = target_scaler.inverse_transform(y_test)
    
    # Evaluate the model
    mae, rmse, r2 = evaluate_model(y_test_true, predicted)
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"R-squared: {r2:.2f}")
    
    # Save the model
    model.save("models/lstm_stock_model.h5")
    print("Model saved successfully.")