from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from fredapi import Fred
import pandas_market_calendars as mcal
from sklearn.preprocessing import MinMaxScaler
from flask import render_template_string
import os

fred = Fred(api_key=os.getenv('FRED_API_KEY'))

nyse = mcal.get_calendar('NYSE')

# --- Data update function ---
def update_data():
    fred = Fred(api_key=os.getenv('FRED_API_KEY'))
    series_dict = {
        'SP500': 'SP500',
        'VIXCLS': 'VIXCLS',
        'DJIA': 'DJIA',
        'HY_BOND_IDX': 'BAMLCC4A0710YTRIV'
    }
    dfs = []
    for col, fred_code in series_dict.items():
        s = fred.get_series(fred_code, observation_start='2020-01-01')
        df = pd.DataFrame(s, columns=[col])
        df['Date'] = df.index
        dfs.append(df)
    # Merge all on 'Date'
    from functools import reduce
    df_data = reduce(lambda left, right: pd.merge(left, right, on='Date', how='left'), dfs)
    df_data['Date'] = pd.to_datetime(df_data['Date'])
    df_data.set_index('Date', inplace=True)
    df_data = df_data.dropna()
    # Filter to NYSE trading days
    schedule = nyse.schedule(start_date=df_data.index.min(), end_date=df_data.index.max())
    df_data = df_data[df_data.index.isin(schedule.index)]
    return df_data

df_data = update_data()

# --- Load model and scaler ---
model = tf.keras.models.load_model('lstm_model.keras')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

SEQ_LEN = 10
N_FEATURES = model.input_shape[-1]

# --- Update data and generate forecasts at startup ---
df_data = update_data()
scaled_data = scaler.transform(df_data)
# Prepare sequences, get latest sequence for prediction
X_latest = scaled_data[-SEQ_LEN:]
X_latest = X_latest.reshape(1, SEQ_LEN, N_FEATURES)
forecast_scaled = model.predict(X_latest)
forecast = scaler.inverse_transform(forecast_scaled)[0]
latest_forecast = dict(zip(df_data.columns, forecast))

# --- Flask app ---
app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def get_latest_forecast():
    # Convert all values to native Python floats for JSON serialization
    forecast_serializable = {k: float(v) for k, v in latest_forecast.items()}
    return jsonify({'forecast': forecast_serializable})

@app.route('/predict', methods=['POST'])
def predict_custom():
    try:
        input_data = request.json['input_data']
        arr = np.array(input_data)
        if arr.shape != (SEQ_LEN, N_FEATURES):
            return jsonify({'error': f'Input shape must be ({SEQ_LEN}, {N_FEATURES})'}), 400
        scaled = scaler.transform(arr)
        pred = model.predict(scaled.reshape(1, SEQ_LEN, N_FEATURES))
        forecast = scaler.inverse_transform(pred).tolist()[0]
        forecast_dict = {k: float(v) for k, v in zip(df_data.columns, forecast)}
        return jsonify({'forecast': forecast_dict})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/plot')
def plot():
    # Prepare table rows from latest_forecast
    table_rows = "".join(
        f"<tr><td>{k}</td><td>{v:.4f}</td></tr>" for k, v in latest_forecast.items()
    )
    html = f"""
    <html>
        <head>
            <title>LSTM Forecast Plot</title>
        </head>
        <body>
            <h2>Multivariate Time-Series Forecasting using LSTM</h2>
            <img src="/static/forecast_plot.png" alt="Forecast Plot" style="max-width:800px;"><br><br>
            <h3>Latest Forecast</h3>
            <table border="1" cellpadding="5">
                <tr><th>Series</th><th>Forecast Value</th></tr>
                {table_rows}
            </table>
        </body>
    </html>
    """
    return render_template_string(html)

@app.route('/metrics')
def metrics():
    metrics_df = pd.read_csv('static/model_metrics.csv')
    table_html = metrics_df.to_html(index=True, float_format="%.4f", classes="table table-striped")
    html = f"""
    <html>
        <head>
            <title>LSTM Model Metrics</title>
            <style>
                .table {{ width: 50%; margin: 20px auto; border-collapse: collapse; }}
                .table th, .table td {{ border: 1px solid #ccc; padding: 8px; text-align: center; }}
                .table th {{ background: #eee; }}
            </style>
        </head>
        <body>
            <img src="/static/loss_curve.png" alt="Loss Curve" style="max-width:600px;">
            <h2 style="text-align:center;">Model Performance Metrics</h2>
            {table_html}
        </body>
    </html>
    """
    return render_template_string(html)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)