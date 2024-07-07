import pandas as pd
import ta
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose


def main(ticker_symbol, seasonal_period):
    print("Preprocessing and analysis of CSV data...")
    try:
        data_file = f"./data/stock/{ticker_symbol}.csv"
        df = pd.read_csv(data_file)
    except:
        return None

    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)

    df["SMA"] = ta.trend.sma_indicator(df["Close"], window=14)
    df["RSI"] = ta.momentum.rsi(df["Close"], window=14)
    df["MACD"] = ta.trend.macd_diff(df["Close"], window_slow=26, window_fast=12, window_sign=9)
    df_bollinger = ta.volatility.BollingerBands(df["Close"], window=20)
    df["upper_band"] = df_bollinger.bollinger_hband()
    df["middle_band"] = df_bollinger.bollinger_mavg()
    df["lower_band"] = df_bollinger.bollinger_lband()
    df["aroon_up"] = ta.trend.aroon_up(df["Close"], df["Low"], window=25)
    df["aroon_down"] = ta.trend.aroon_down(df["Close"], df["Low"], window=25)

    open_prices = df["Open"]
    close_prices = df["Close"]

    kicking_pattern = np.zeros_like(open_prices)

    for i in range(1, len(open_prices)):
        if open_prices.iloc[i] < open_prices.iloc[i-1] and \
        open_prices.iloc[i] > close_prices.iloc[i-1] and \
        close_prices.iloc[i] > open_prices.iloc[i-1] and \
        close_prices.iloc[i] < close_prices.iloc[i-1] and \
        open_prices.iloc[i] - close_prices.iloc[i] > open_prices.iloc[i-1] - close_prices.iloc[i-1]:
            kicking_pattern[i] = 100
        
    
    df["kicking"] = kicking_pattern

    def calculate_atr(high, low, close, window=14):
        true_ranges = np.maximum.reduce([high - low, np.abs(high - close.shift()), np.abs(low - close.shift())])
        atr = np.zeros_like(high)
        atr[window - 1] = np.mean(true_ranges[:window])
        for i in range(window, len(high)):
            atr[i] = (atr[i - 1] * (window - 1) + true_ranges[i]) / window
        return atr

    df["ATR"] = calculate_atr(df["High"], df["Low"], df["Close"], window=14)

    df["upper_band_supertrend"] = df["High"] - (df["ATR"])
    df["lower_band_supertrend"] = df["Low"] + (df["ATR"])

    uptrend_conditions = [
        (df["Close"] > df["lower_band_supertrend"]),
        (df["Close"] > df["SMA"]),
        (df["Close"] > df["middle_band"]),
        (df["Close"] > df["MACD"]),
        (df["RSI"] > 50),
        (df["aroon_up"] > df["aroon_down"]),
        (df["kicking"] == 1),
        (df["Close"] > df["upper_band_supertrend"])
    ]

    downtrend_conditions = [
        (df["Close"] < df["upper_band_supertrend"]),
        (df["Close"] < df["SMA"]),
        (df["Close"] < df["middle_band"]),
        (df["Close"] < df["MACD"]),
        (df["RSI"] < 50),
        (df["aroon_up"] < df["aroon_down"]),
        (df["kicking"] == -1),
        (df["Close"] < df["lower_band_supertrend"])
    ]

    df["supertrend_signal"] = 0

    df.loc[np.any(uptrend_conditions, axis=0), "supertrend_signal"] = 1
    df.loc[np.any(downtrend_conditions, axis=0), "supertrend_signal"] = -1

    try:
        result = seasonal_decompose(df["Close"], model="additive", period=seasonal_period)
    except ValueError as e:
        print(f"Error during seasonal decomposition: {e}")
        return None

    df["trend"] = result.trend
    df["seasonal"] = result.seasonal
    df["residual"] = result.resid

    df2 = pd.concat([df[col] for col in ["Close", "Open", "Adj Close", "Volume", "High", "Low", "SMA", 
                                         "MACD", "upper_band", "middle_band", "lower_band", "supertrend_signal", 
                                         "RSI", "aroon_up", "aroon_down", "kicking", "upper_band_supertrend", 
                                         "lower_band_supertrend", "trend", "seasonal", "residual"]], axis=1)

    df2.fillna(0, inplace=True)

    df2.to_csv(f"./data/preprocessed/{ticker_symbol}.csv")
    print(f"The data is uploaded and saved in ./data/preprocessed/{ticker_symbol}.csv")

    signal_changes = df["supertrend_signal"].diff().fillna(0)
    consecutive_mask = (signal_changes == 0) & (signal_changes.shift(-1) == 0)
    df.loc[consecutive_mask, "supertrend_signal"] = 0

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(12, 8), sharex=True)

    ax1.plot(df.index, df["Open"], label="Open")
    ax1.plot(df.index, df["Close"], label="Close")
    ax1.plot(df.index, df["trend"], label="Trend")

    ax1.plot(df.index, df["SMA"], label="SMA")
    ax1.fill_between(
        df.index, df["upper_band"], df["lower_band"], alpha=0.2, color="gray"
    )
    ax1.plot(df.index, df["upper_band"], linestyle="dashed", color="gray")
    ax1.plot(df.index, df["middle_band"], linestyle="dashed", color="gray")
    ax1.plot(df.index, df["lower_band"], linestyle="dashed", color="gray")
    ax1.scatter(
        df.index[df["supertrend_signal"] == 1],
        df["Close"][df["supertrend_signal"] == 1],
        marker="^",
        color="green",
        s=100,
    )
    ax1.scatter(
        df.index[df["supertrend_signal"] == -1],
        df["Close"][df["supertrend_signal"] == -1],
        marker="v",
        color="red",
        s=100,
    )
    ax1.legend()

    ax2.plot(df.index, df["aroon_up"], label="Aroon Up")
    ax2.plot(df.index, df["aroon_down"], label="Aroon Down")
    ax2.legend()

    ax3.plot(df.index, df["RSI"], label="RSI")
    ax3.legend()

    ax4.plot(df.index, df["seasonal"], label="Seasonal")
    ax4.legend()

    ax5.plot(df.index, df["residual"], label="Residual")
    ax5.legend()

    plt.xlim(df.index[0], df.index[-1])

    plt.show()
