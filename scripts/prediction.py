import pandas as pd
import numpy as np
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import r2_score
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

def prediction(ticker_symbol, data_file, model_file, output_file, forecast_days, timesteps):
    try:
        df = pd.read_csv(data_file)
    except Exception as e:
        print(f"Error reading the data file: {e}")
        return
    
    df['Date'] = pd.to_datetime(df['Date'])
    
    try:
        model = load_model(model_file)
    except Exception as e:
        print(f"Error loading the model: {e}")
        return

    scaler = MaxAbsScaler()
    data_norm = scaler.fit_transform(df[['Close']])

    def create_sequences(data, timesteps):
        X = []
        for i in range(timesteps, len(data)):
            X.append(data[i - timesteps:i])
        return np.array(X)

    X_data = create_sequences(data_norm, timesteps)

    predictions = model.predict(X_data)

    predicted_values = scaler.inverse_transform(predictions)

    forecast_input = data_norm[-timesteps:].tolist()

    for _ in range(forecast_days):
        next_input = np.array(forecast_input[-timesteps:]).reshape(1, timesteps, 1)
        next_pred = model.predict(next_input)
        forecast_input.append(next_pred[0])

    future_predictions = scaler.inverse_transform(np.array(forecast_input[-forecast_days:]).reshape(-1, 1)).flatten()

    future_dates = pd.date_range(start=df['Date'].iloc[-1], periods=forecast_days+1)
    forecast_df = pd.DataFrame({'Date': future_dates[1:], 'Predicted Close': future_predictions})

    df['Predicted Close'] = np.concatenate((np.full(timesteps, np.nan), predicted_values.flatten()), axis=0)

    actual_values = df['Close'].values[timesteps:]
    predicted_values = df['Predicted Close'].values[timesteps:]
    r2 = r2_score(actual_values, predicted_values)
    print(f"Coefficient of determination (R^2): {r2}")

    result_df = pd.concat([df, forecast_df], ignore_index=True)
    result_df.to_csv(output_file, index=False)

    print(f"Predictions are saved to a file: {output_file}")

    plt.figure(figsize=(14, 7))
    plt.plot(df['Date'], df['Close'], label='Actual Close', color='blue')
    plt.plot(df['Date'], df['Predicted Close'], label='Predicted Close', color='red')
    plt.plot(forecast_df['Date'], forecast_df['Predicted Close'], label='Forecast Close', linestyle='--', color='green')
    plt.title('Actual vs Predicted Stock Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"./data/prediction/{ticker_symbol}.png")
    plt.show()

def main(ticker_symbol, model_name, forecast_days, timesteps):
    try:
        data_file_path = f"./data/preprocessed/{ticker_symbol}.csv"
        data_file = pd.read_csv(data_file_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    model_file = f"./models/{model_name}.keras"
    output_file = f"./data/prediction/{ticker_symbol}.csv"
    
    prediction(ticker_symbol, data_file_path, model_file, output_file, forecast_days, timesteps)