import yfinance as yf
import pandas as pd


def get_ticker_info(ticker_symbol):
    tic = yf.Ticker(ticker_symbol)
    info = tic.get_info()
    return info

def download_data(ticker_symbol, interval):
    print(f"Uploading data for {ticker_symbol} from Yahoo Finance...")
    data = yf.download(ticker_symbol, period="max", interval=f"{interval}")
    df = pd.DataFrame(data)
    return df

def save_data(dataframe, ticker_symbol):
    data_file = f"./data/stock/{ticker_symbol}.csv"
    dataframe.to_csv(data_file)
    print("The data is uploaded and saved in", data_file)

def main(ticker_symbol, interval):
    ticker_info = get_ticker_info(ticker_symbol)

    print("Ticker:", ticker_info["shortName"])

    download = input("Do you want to download this ticker?: ").lower()

    if download == "y" or download == "yes":
        df = download_data(ticker_symbol, interval)
        save_data(df, ticker_symbol)
    else:
        print("Exiting Script..")