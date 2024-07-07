# StockSage
StockSage is a machine learning project designed to forecast stock prices using historical market data. The project utilizes several Python libraries and machine learning models to analyze and predict stock market trends.

## Features

- **Data Collection**: Retrieve historical stock data from Yahoo Finance.
- **Data Preprocessing**: Clean and prepare data for analysis.

![Figure_1](https://github.com/playaaaa/StockSage/assets/174910162/b96fe684-2f8c-40e4-96bb-53298e5d8130)

- **Model Training**: Train machine learning models using scikit-learn and TensorFlow.
- **Prediction**: Generate stock price predictions.
- **Visualization**: Visualize stock price trends and prediction results.

![Figure_2](https://github.com/playaaaa/StockSage/assets/174910162/e109a8f7-268f-4cd0-82ef-ea43ceb4c0bd)


## Installation

1. **Clone the repository:**

  ```bash
  git clone https://github.com/playaaaa/StockSage.git
  cd StockSage
  ```

## Requirements
- Python 3.11 (you can install it [here](https://www.python.org/downloads/release/python-3110/))

Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Configure the project:**

    Update the configuration in `config/config.py`:

    ```python
    ticker_symbol = 'AAPL'
    interval = '1d'
    timesteps = 60
    model_name = 'lstm_model'
    seasonal_period = 12
    epochs = 50
    batch_size = 32
    forecast_days = 30
    ```

2. **Run the main script:**

    ```bash
    python main.py
    ```

3. **Interact with the main menu:**

    Choose options from the main menu to perform various actions such as installing dependencies, downloading data, training models, etc.

## Important Notes
We use [Yahoo Finance](https://finance.yahoo.com) for the financial data. Please get tickers only from here.

---

Thank you for using StockSage!

Ton wallet for donations : UQDnc8jpNmoToTQbs51YfgMSVu4Ro4qJwV9oow76Kq9xxvhJ
