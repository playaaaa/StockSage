import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error
import matplotlib.pyplot as plt


def load_data(file_path: str) -> pd.DataFrame:
    data = pd.read_csv(file_path)
    return data[['Close']]  # Загружаем только столбец 'Close'

def normalize_data(data: pd.DataFrame) -> np.ndarray:
    scaler = MaxAbsScaler()
    return scaler.fit_transform(data), scaler

def create_sequences(data: np.ndarray, timesteps: int):
    X = []
    y = []
    for i in range(timesteps, len(data)):
        X.append(data[i - timesteps:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def get_reward(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    reward = (((1 - mape) * 0.1) + ((r2) * 1.9)) / 2
    return reward

def evaluate_model(model: tf.keras.Model, X_test: np.ndarray, y_test: np.ndarray):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    reward = get_reward(y_test, y_pred)
    return mse, mape, r2, reward, y_pred

def plot_predictions(y_test: np.ndarray, y_pred: np.ndarray):
    plt.plot(y_test, label="Actual")
    plt.plot(y_pred, label="Predicted")
    plt.legend()
    plt.show()

def main(ticker_symbol, timesteps, model_name):
    try:
        data = load_data(f"./data/preprocessed/{ticker_symbol}.csv")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    test_data = data.iloc[int(0.8 * len(data)):]
    test_data_norm, scaler = normalize_data(test_data)
    
    X_test, y_test = create_sequences(test_data_norm, timesteps)
    
    model = load_model(f"./models/{model_name}.keras")
    print(f"\nEvaluating Model: {model_name}...")
    
    mse, mape, r2, reward, y_pred = evaluate_model(model, X_test, y_test)
    
    print("MAPE:", mape)
    print("MSE:", mse)
    print("R2:", r2)
    print("Reward:", reward)
    
    plot_predictions(y_test, y_pred)
