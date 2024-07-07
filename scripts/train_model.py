import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, PReLU
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error


def load_data(file_path: str) -> pd.DataFrame:
    data = pd.read_csv(file_path)
    return data[['Close']]

def get_reward(y_true, y_pred):
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    reward = (((1 - mape) * 0.1) + ((r2) * 1.9)) / 2
    return reward

def create_LSTM_model() -> Sequential:
    model = Sequential()
    
    model.add(LSTM(units=150, return_sequences=True))
    model.add(PReLU())
    model.add(PReLU())
    model.add(PReLU())
    model.add(LSTM(units=150))
    model.add(PReLU())
    model.add(PReLU())
    model.add(PReLU())

    model.add(Dense(units=1, activation='linear'))

    return model

def create_sequences(data, timesteps):
    X = []
    y = []
    for i in range(timesteps, len(data)):
        X.append(data[i-timesteps:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def main(ticker_symbol, timesteps, epochs, batch_size, model_name):
    try:
        data = load_data(f"./data/preprocessed/{ticker_symbol}.csv")
    except:
        return None

    train_data = data.iloc[:int(0.8*len(data))]
    test_data = data.iloc[int(0.8*len(data)):]

    scaler = MaxAbsScaler()
    train_data_norm = scaler.fit_transform(train_data)
    test_data_norm = scaler.transform(test_data)

    X_train, y_train = create_sequences(train_data_norm, timesteps)
    X_test, y_test = create_sequences(test_data_norm, timesteps)

    model = create_LSTM_model()

    model.compile(optimizer='adam', loss="huber")

    for i in range(epochs):
        print(f"Epoch {i+1} / {epochs}")
        history = model.fit(X_train, y_train, batch_size=batch_size, validation_data=(X_test, y_test), epochs=1)
        
        y_pred_test = model.predict(X_test)
        test_reward = get_reward(y_test, y_pred_test)
        
        print("Reward on the test:", test_reward)
        
        if i == 0:
            best_reward = test_reward
        
        if test_reward >= best_reward:
            best_reward = test_reward
            print("The model is saved!")
            model.save(f"./models/{model_name}.keras")

    model = tf.keras.models.load_model(f"./models/{model_name}.keras")
    y_pred_test = model.predict(X_test)
    test_reward = get_reward(y_test, y_pred_test)
    test_loss = model.evaluate(X_test, y_test)

    print("The final reward on the test:", test_reward)
    print("The final loss on the test:", test_loss)
