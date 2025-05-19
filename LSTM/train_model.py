from fredapi import Fred
import pandas as pd
import statsmodels.api as sm
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import numpy as np
import pandas_market_calendars as mcal
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify

nyse = mcal.get_calendar('NYSE')

def update_fred_data():
    # Update FRED data
    # This function is a placeholder. You can implement it to fetch and update data from FRED.
    fred = Fred(api_key='bcfc73a6d0f3e505e8481721b9ab6f4e')
    sp_500 = fred.get_series('SP500', observation_start='2020-01-01')
    vix = fred.get_series('VIXCLS', observation_start='2020-01-01')
    djia = fred.get_series('DJIA', observation_start='2020-01-01')
    bond = fred.get_series('BAMLCC4A0710YTRIV', observation_start='2020-01-01')

    df_sp500 = pd.DataFrame(sp_500, columns=['SP500'])
    df_sp500['Date'] = df_sp500.index

    df_vix = pd.DataFrame(vix, columns=['VIXCLS'])
    df_vix['Date'] = df_vix.index

    df_djia = pd.DataFrame(djia, columns=['DJIA'])
    df_djia['Date'] = df_djia.index

    df_bond = pd.DataFrame(bond, columns=['BAMLCC4A0710YTRIV'])
    df_bond['Date'] = df_bond.index
    df_bond = df_bond.rename(columns={'BAMLCC4A0710YTRIV': 'HY_BOND_IDX'})

    df_data = df_sp500.merge(df_vix, on='Date', how='left')
    df_data = df_data.merge(df_djia, on='Date', how='left')
    df_data = df_data.merge(df_bond, on='Date', how='left')
    df_data['Date'] = pd.to_datetime(df_data['Date'])
    df_data.set_index('Date', inplace=True)
    df_data = df_data.dropna()
    schedule = nyse.schedule(start_date=df_data.index.min(), end_date=df_data.index.max())
    df_data = df_data[df_data.index.isin(schedule.index)]
    return df_data

df_data = update_fred_data()

# Normalize data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_data)

# Prepare sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

SEQ_LEN = 10
X, y = create_sequences(scaled_data, SEQ_LEN)

# Split into train/test
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(SEQ_LEN, df_data.shape[1])),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(df_data.shape[1])
])
model.compile(optimizer='adam', loss='mean_squared_error')

# Train model and capture history
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=16,
    validation_data=(X_test, y_test)
)

# Save training and validation loss to CSV
history_df = pd.DataFrame(history.history)
history_df.to_csv('static/model_history.csv', index=False)

plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training & Validation Loss')
plt.legend()
plt.savefig('static/loss_curve.png')
plt.close()

# Number of days you are forecasting
n_days = 5  # or whatever number you want

# Get the last date in your df_data
last_date = df_data.index[-1]

# Use NYSE calendar to get the next n_days business days
future_dates = nyse.valid_days(start_date=last_date + pd.Timedelta(days=1), end_date=last_date + pd.Timedelta(days=30))
future_dates = future_dates[:n_days]

# Predict
y_pred = model.predict(X_test)
y_pred_rescaled = scaler.inverse_transform(y_pred)
y_test_rescaled = scaler.inverse_transform(y_test)

# Convert predictions to DataFrame with correct column names
pred_df = pd.DataFrame(y_pred_rescaled, columns=df_data.columns)
pd.set_option('display.float_format', '{:.4f}'.format)

metrics = {}
for i, col in enumerate(df_data.columns):
    mse = mean_squared_error(y_test_rescaled[:, i], y_pred_rescaled[:, i])
    mae = mean_absolute_error(y_test_rescaled[:, i], y_pred_rescaled[:, i])
    r2 = r2_score(y_test_rescaled[:, i], y_pred_rescaled[:, i])
    metrics[col] = {'MSE': mse, 'MAE': mae, 'R2': r2}

metrics_df = pd.DataFrame(metrics).T
metrics_df.to_csv('static/model_metrics.csv')

def forecast_n_days(model, scaler, df_data, n_days=5):
    seq = scaler.transform(df_data)[-SEQ_LEN:].copy()
    forecasts = []
    for _ in range(n_days):
        pred_scaled = model.predict(seq.reshape(1, SEQ_LEN, df_data.shape[1]))
        pred = scaler.inverse_transform(pred_scaled)[0]
        forecasts.append(pred)
        seq = np.vstack([seq[1:], scaler.transform(pred.reshape(1, -1))])
    return np.array(forecasts)

# Generate n_days forecasts for future dates
forecasts = forecast_n_days(model, scaler, df_data, n_days=n_days)
forecast_df = pd.DataFrame(forecasts, columns=df_data.columns, index=future_dates)
forecast_df = forecast_df.reset_index().rename(columns={'index': 'Date'})

# Add a column to distinguish actuals vs. forecast
df_data_with_flag = df_data.copy()
df_data_with_flag['Type'] = 'Actual'
forecast_df_with_flag = forecast_df.copy()
forecast_df_with_flag['Type'] = 'Forecast'
forecast_df_with_flag = forecast_df_with_flag.set_index('Date')

# Concatenate actuals and forecasts
combined_df = pd.concat([df_data_with_flag, forecast_df_with_flag], axis=0)

# Reset index for plotting if needed
combined_df = combined_df.reset_index().rename(columns={'index': 'Date'})

fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
axes = axes.flatten()

for i, col in enumerate(df_data.columns):
    # Plot actuals
    actuals = combined_df[combined_df['Type'] == 'Actual']
    axes[i].plot(actuals['Date'], actuals[col], label='Actual (History)', color='blue')
    # Plot forecasts
    forecasts = combined_df[combined_df['Type'] == 'Forecast']
    axes[i].plot(forecasts['Date'], forecasts[col], label='Forecast', color='orange', linestyle='--')
    axes[i].set_title(f'{col}: Actual & Forecast')
    axes[i].set_xlabel('Date')
    axes[i].set_ylabel(col)
    axes[i].legend()

fig.suptitle('Multivariate Time-Series Forecasting using LSTM', fontsize=18)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('static/forecast_plot.png')
plt.show()

# Save the model
model.save('lstm_model.keras')

# Save the scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

