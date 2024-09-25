# trade_volume_prediction_gui.py

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime, timedelta

# ----------------------------
# Global Variables
# ----------------------------

# To store the trained model and data
model = None
scaler = None
data_values = None
sequence_length = None
ticker_symbol = None
start_date = None
end_date = None
data = None  # Add this line to declare 'data' as a global variable

# ----------------------------
# GUI Functions
# ----------------------------

def start_training():
    global model, scaler, data_values, sequence_length, ticker_symbol, start_date, end_date, data  # Include 'data' here

    # Get user inputs
    ticker_symbol = stock_entry.get().upper()
    start_date = start_date_entry.get()
    end_date = end_date_entry.get()
    sequence_length = int(sequence_entry.get())
    epochs = int(epochs_entry.get())
    batch_size = int(batch_size_entry.get())

    if not ticker_symbol:
        messagebox.showerror("Input Error", "Please enter a stock ticker symbol.")
        return

    # Data Collection
    try:
        data = yf.download(ticker_symbol, start=start_date, end=end_date)
        if data.empty:
            messagebox.showerror("Data Error", "No data found for the specified dates.")
            return
    except Exception as e:
        messagebox.showerror("Data Error", f"Error downloading data: {e}")
        return

    # Data Preprocessing
    data.reset_index(inplace=True)
    data.fillna(method='ffill', inplace=True)
    data.set_index('Date', inplace=True)

    # Feature Engineering
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    data_features = data[features]

    # Scaling
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_features)

    data_values = scaled_data

    # Create sequences
    X, y = create_sequences(data_values, sequence_length)

    if len(X) == 0:
        messagebox.showerror("Data Error", "Not enough data to create sequences. Adjust the sequence length.")
        return

    # Model Building
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.3))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(32))
    model.add(Dropout(0.3))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mae')  # Changed loss to Mean Absolute Error

    # Model Training
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=1)

    # Store history for plotting
    model.history = history
    model.X_test = X_test
    model.y_test = y_test

    messagebox.showinfo("Training Complete", "Model training is complete!")

def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, 4])  # Predicting the 'Volume' column
    return np.array(X), np.array(y)

def plot_volume():
    if data_values is None:
        messagebox.showerror("Data Error", "Please train the model first.")
        return

    # Plot the trade volume over time
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data['Volume'], label='Trade Volume')
    plt.title(f'Historical Trade Volume for {ticker_symbol}')
    plt.xlabel('Date')
    plt.ylabel('Trade Volume')
    plt.legend()
    plt.show()

def plot_loss():
    if model is None:
        messagebox.showerror("Model Error", "Please train the model first.")
        return

    history = model.history

    # Determine the correct keys for loss
    if 'loss' in history.history:
        loss_key = 'loss'
        val_loss_key = 'val_loss'
    elif 'mae' in history.history:
        loss_key = 'mae'
        val_loss_key = 'val_mae'
    else:
        messagebox.showerror("History Error", "Loss keys not found in history.")
        return

    # Plot training and validation loss
    plt.figure(figsize=(14, 5))
    plt.plot(history.history[loss_key], label='Training Loss')
    plt.plot(history.history[val_loss_key], label='Validation Loss')
    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def plot_predictions():
    if model is None:
        messagebox.showerror("Model Error", "Please train the model first.")
        return

    X_test = model.X_test
    y_test = model.y_test

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Inverse transform the predictions
    y_test_inverse = scaler.inverse_transform(np.concatenate((np.zeros((len(y_test), 4)), y_test.reshape(-1, 1)), axis=1))[:, 4]
    y_pred_inverse = scaler.inverse_transform(np.concatenate((np.zeros((len(y_pred), 4)), y_pred), axis=1))[:, 4]

    # Plot actual vs. predicted volumes
    plt.figure(figsize=(14, 7))
    plt.plot(y_test_inverse, label='Actual Volume')
    plt.plot(y_pred_inverse, label='Predicted Volume')
    plt.title('Actual vs Predicted Trade Volume')
    plt.xlabel('Time')
    plt.ylabel('Trade Volume')
    plt.legend()
    plt.show()

def predict_future():
    if model is None:
        messagebox.showerror("Model Error", "Please train the model first.")
        return

    # Get the number of days to predict
    try:
        future_days = int(future_days_entry.get())
        if future_days <= 0:
            raise ValueError
    except ValueError:
        messagebox.showerror("Input Error", "Please enter a valid number of future days to predict.")
        return

    # Get the last 'sequence_length' values from the dataset
    last_sequence = data_values[-sequence_length:]
    predicted_volumes = []

    # Predict future volumes
    for _ in range(future_days):
        last_sequence_reshaped = last_sequence.reshape((1, sequence_length, data_values.shape[1]))
        next_volume_scaled = model.predict(last_sequence_reshaped)
        # Prepare input for inverse transformation
        next_volume_full = np.concatenate((np.zeros((1, 4)), next_volume_scaled), axis=1)
        next_volume = scaler.inverse_transform(next_volume_full)[0, 4]
        predicted_volumes.append(next_volume)
        # Update the last_sequence with the new predicted value
        next_sequence = np.concatenate((last_sequence[1:], np.concatenate((np.zeros((1, 4)), next_volume_scaled), axis=1)), axis=0)
        last_sequence = next_sequence

    # Prepare dates for plotting
    last_date = data.index[-1]
    future_dates = [last_date + timedelta(days=i+1) for i in range(future_days)]

    # Adjust dates to skip weekends
    future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=future_days)

    # Plot the future predictions
    plt.figure(figsize=(14, 7))
    # Plot historical data
    historical_volumes = data['Volume'].values
    historical_dates = data.index
    plt.plot(historical_dates, historical_volumes, label='Historical Volume')
    # Plot future predictions
    plt.plot(future_dates, predicted_volumes, label='Future Predicted Volume', color='red')
    plt.title(f'Future Trade Volume Prediction for {ticker_symbol}')
    plt.xlabel('Date')
    plt.ylabel('Trade Volume')
    plt.legend()
    plt.show()

    # Display the predicted values
    predictions_text = "\n".join([f"{date.date()}: {volume:.2f}" for date, volume in zip(future_dates, predicted_volumes)])
    messagebox.showinfo("Future Predictions", f"Predicted Trade Volumes:\n\n{predictions_text}")

# ----------------------------
# GUI Setup
# ----------------------------

root = tk.Tk()
root.title("Stock Trade Volume Prediction")

# Stock Selection Frame
stock_frame = ttk.LabelFrame(root, text="Stock Selection")
stock_frame.grid(column=0, row=0, padx=10, pady=10, sticky="W")

ttk.Label(stock_frame, text="Stock Ticker Symbol:").grid(column=0, row=0, padx=5, pady=5, sticky="E")
stock_entry = ttk.Entry(stock_frame, width=15)
stock_entry.grid(column=1, row=0, padx=5, pady=5)

ttk.Label(stock_frame, text="Start Date (YYYY-MM-DD):").grid(column=0, row=1, padx=5, pady=5, sticky="E")
start_date_entry = ttk.Entry(stock_frame, width=15)
start_date_entry.insert(0, "2015-01-01")
start_date_entry.grid(column=1, row=1, padx=5, pady=5)

ttk.Label(stock_frame, text="End Date (YYYY-MM-DD):").grid(column=0, row=2, padx=5, pady=5, sticky="E")
end_date_entry = ttk.Entry(stock_frame, width=15)
end_date_entry.insert(0, datetime.today().strftime('%Y-%m-%d'))
end_date_entry.grid(column=1, row=2, padx=5, pady=5)

# Parameters Frame
params_frame = ttk.LabelFrame(root, text="Model Parameters")
params_frame.grid(column=0, row=1, padx=10, pady=10, sticky="W")

ttk.Label(params_frame, text="Sequence Length:").grid(column=0, row=0, padx=5, pady=5, sticky="E")
sequence_entry = ttk.Entry(params_frame, width=10)
sequence_entry.insert(0, "60")
sequence_entry.grid(column=1, row=0, padx=5, pady=5)

ttk.Label(params_frame, text="Epochs:").grid(column=0, row=1, padx=5, pady=5, sticky="E")
epochs_entry = ttk.Entry(params_frame, width=10)
epochs_entry.insert(0, "50")
epochs_entry.grid(column=1, row=1, padx=5, pady=5)

ttk.Label(params_frame, text="Batch Size:").grid(column=0, row=2, padx=5, pady=5, sticky="E")
batch_size_entry = ttk.Entry(params_frame, width=10)
batch_size_entry.insert(0, "32")
batch_size_entry.grid(column=1, row=2, padx=5, pady=5)

ttk.Label(params_frame, text="Future Days to Predict:").grid(column=0, row=3, padx=5, pady=5, sticky="E")
future_days_entry = ttk.Entry(params_frame, width=10)
future_days_entry.insert(0, "5")
future_days_entry.grid(column=1, row=3, padx=5, pady=5)

# Buttons Frame
buttons_frame = ttk.Frame(root)
buttons_frame.grid(column=0, row=2, padx=10, pady=10, sticky="W")

train_button = ttk.Button(buttons_frame, text="Start Training", command=start_training)
train_button.grid(column=0, row=0, padx=5, pady=5)

plot_volume_button = ttk.Button(buttons_frame, text="Plot Volume", command=plot_volume)
plot_volume_button.grid(column=1, row=0, padx=5, pady=5)

plot_loss_button = ttk.Button(buttons_frame, text="Plot Loss", command=plot_loss)
plot_loss_button.grid(column=2, row=0, padx=5, pady=5)

plot_predictions_button = ttk.Button(buttons_frame, text="Plot Predictions", command=plot_predictions)
plot_predictions_button.grid(column=3, row=0, padx=5, pady=5)

predict_future_button = ttk.Button(buttons_frame, text="Predict Future Volume", command=predict_future)
predict_future_button.grid(column=4, row=0, padx=5, pady=5)

# Start the GUI event loop
root.mainloop()