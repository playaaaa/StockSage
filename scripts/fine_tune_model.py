import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import signal


def get_reward(y_true, y_pred):
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    reward = (((1 - mape) * 0.1) + (r2 * 1.9)) / 2
    return reward

def load_data(ticker_symbol):
    try:
        data = pd.read_csv(f"./data/preprocessed/{ticker_symbol}.csv")
        return data
    except:
        return None

def preprocess_data(data):
    scaler = MaxAbsScaler()
    data_norm = scaler.fit_transform(data[["Close"]])
    return data_norm

def create_sequences(data, timesteps):
    X = []
    y = []
    for i in range(timesteps, len(data)):
        X.append(data[i - timesteps : i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def handle_interrupt():
    print("\nThe interrupt has been received")
    user_input = input("Are you sure you want to complete the program? (yes/no): ")
    if user_input.lower() == "yes":
        exit(0)
    else:
        print("We continue the learning process")

def train_and_evaluate_model(X_train, y_train, X_test, y_test, reward_threshold, model_name):
    rewards = []
    best_reward = 0
    count = 0

    while True:
        model = load_model(f"./models/{model_name}.keras")
        print("\nEvaluation of the model")
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        reward = get_reward(y_test, y_pred)
        rewards.append(reward)
        print("Rewards:", rewards)
        print("MAPE:", mape)
        print("MSE:", mse)
        print("R2:", r2)
        count += 1
        print("Number of cycles: ", count)

        if reward >= reward_threshold:
            print("The reward has been reached!")
            model.save(f"./models/{model_name}.keras")
            break
        else:
            epochs = 10
            print(f"Training a model on {epochs} epochs")
            batch_size = 32
            for i in range(epochs):
                print(f"Epochs {i+1} from {epochs}")
                model.fit(X_train, y_train, batch_size=batch_size, validation_data=(X_test, y_test), epochs=1)
                y_pred_test = model.predict(X_test)
                test_reward = get_reward(y_test, y_pred_test)
                print("The reward on the test:", test_reward)
                if test_reward >= best_reward:
                    print("The model is saved!")
                    best_reward = test_reward
                    model.save(f"./models/{model_name}.keras")
                if test_reward >= reward_threshold:
                    print("The model has reached the reward", test_reward, ". Save and stop learning!")
                    model.save(f"./models/{model_name}.keras")
                    break

def main(ticker_symbol, model_name, timesteps):
    print("Starting train the model...")
    signal.signal(signal.SIGINT, handle_interrupt)

    data = load_data(ticker_symbol)
    if data is None:
        exit(0)

    train_data = data.iloc[:int(0.8 * len(data))]
    test_data = data.iloc[int(0.8 * len(data)):]

    train_data_norm = preprocess_data(train_data)
    test_data_norm = preprocess_data(test_data)

    X_train, y_train = create_sequences(train_data_norm, timesteps)
    X_test, y_test = create_sequences(test_data_norm, timesteps)

    reward_threshold = float(input("Enter the reward (0-1, 0.9 is recommended): "))

    train_and_evaluate_model(X_train, y_train, X_test, y_test, reward_threshold, model_name)
