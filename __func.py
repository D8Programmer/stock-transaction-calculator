import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def load_data(stock_file: str, purchase_file: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load stock prices and user purchase data from CSV files.

    Args:
        stock_file (str): Path to the stock data CSV file.
        purchase_file (str): Path to the user purchase data CSV file.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Stock data and purchase data as DataFrames.
    """
    stock_data = pd.read_csv(stock_file, parse_dates=['time'])
    purchase_data = pd.read_csv(purchase_file, parse_dates=['time'])
    return stock_data, purchase_data


def interpolate_prices(stock_data: pd.DataFrame, target_times: pd.Series) -> pd.Series:
    """
    Interpolate stock prices for target times using linear interpolation.

    Args:
        stock_data (pd.DataFrame): Stock data containing 'time' and 'price'.
        target_times (pd.Series): Timestamps for which to interpolate prices.

    Returns:
        pd.Series: Interpolated prices indexed by target times.
    """
    stock_data = stock_data.sort_values('time')
    target_time_in_ns = target_times.astype('int64')
    stock_time_in_ns = stock_data['time'].astype('int64')

    interp_func = interp1d(
        stock_time_in_ns, stock_data['price'], kind='linear', bounds_error=False, fill_value='extrapolate'
    )

    interpolated_prices = interp_func(target_time_in_ns)
    return pd.Series(interpolated_prices, index=target_times)


def calculate_transaction_details(purchase_data: pd.DataFrame, interpolated_prices: pd.Series) -> pd.DataFrame:
    """
    Calculate user profit/loss and percentage changes for each transaction.

    Args:
        purchase_data (pd.DataFrame): User transaction data containing 'time', 'action', 'quantity'.
        interpolated_prices (pd.Series): Interpolated stock prices indexed by time.

    Returns:
        pd.DataFrame: Detailed transaction data with profit/loss and percentage changes.
    """
    if purchase_data.empty or interpolated_prices.empty:
        raise ValueError("Input data cannot be empty.")

    purchase_data = purchase_data.sort_values('time').copy()
    purchase_data['estimated_price'] = purchase_data['time'].map(interpolated_prices)
    purchase_data['cost'] = purchase_data['quantity'] * purchase_data['estimated_price']

    purchase_data['stock_change_percentage'] = purchase_data['estimated_price'].pct_change().mul(100).fillna(0).round(2)
    purchase_data['user_change_percentage'] = purchase_data['cost'].pct_change().mul(100).fillna(0).round(2)

    return purchase_data


def save_results_to_csv(data: pd.DataFrame, output_file: str) -> None:
    """
    Save transaction details to a CSV file.

    Args:
        data (pd.DataFrame): DataFrame containing transaction details.
        output_file (str): Path to the output CSV file.
    """
    data.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


def plot_stock_with_transactions(stock_data: pd.DataFrame, purchase_data: pd.DataFrame) -> None:
    """
    Plot stock prices and annotate buy/sell transactions.

    Args:
        stock_data (pd.DataFrame): Stock data containing 'time' and 'price'.
        purchase_data (pd.DataFrame): Transaction data containing 'time', 'action', and 'estimated_price'.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(stock_data['time'], stock_data['price'], label='Stock Prices', marker='o', linewidth=2)

    buy_data = purchase_data[purchase_data['action'] == 'BUY']
    sell_data = purchase_data[purchase_data['action'] == 'SELL']

    plt.scatter(buy_data['time'], buy_data['estimated_price'], color='green', label='Buy', marker='o', s=50, zorder=2)
    plt.scatter(sell_data['time'], sell_data['estimated_price'], color='red', label='Sell', marker='o', s=50, zorder=2)

    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title('Stock Prices with Transactions')
    plt.legend()
    plt.grid()
    plt.show()