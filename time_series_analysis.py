import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from data_preprocessing import load_data, preprocess_data, get_top_products

def prepare_time_series(transactions, stock_code):
    product_data = transactions[transactions['StockCode'] == stock_code]
    weekly_data = product_data.set_index('TransactionDate').resample('W')['Quantity'].sum().reset_index()
    weekly_data.set_index('TransactionDate', inplace=True)
    return weekly_data

def analyze_time_series(data, stock_code):
    plt.figure(figsize=(12, 6))
    data['Quantity'].plot()
    plt.title(f'Weekly Sales for Stock Code {stock_code}')
    plt.xlabel('Date')
    plt.ylabel('Quantity Sold')
    plt.savefig(f'../output/time_series_{stock_code}.png')
    plt.close()

    decomposition = seasonal_decompose(data['Quantity'], model='additive', period=52)
    decomposition.plot()
    plt.tight_layout()
    plt.savefig(f'../output/decomposition_{stock_code}.png')
    plt.close()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(data['Quantity'], ax=ax1, lags=52)
    plot_pacf(data['Quantity'], ax=ax2, lags=52)
    plt.tight_layout()
    plt.savefig(f'../output/acf_pacf_{stock_code}.png')
    plt.close()

if __name__ == "__main__":
    transactions_1, transactions_2, customer_demographics, product_info = load_data()
    transactions, customer_demographics, product_info = preprocess_data(transactions_1, transactions_2, customer_demographics, product_info)
    top_10_quantity, _ = get_top_products(transactions)

    for stock_code in top_10_quantity.index:
        time_series_data = prepare_time_series(transactions, stock_code)
        analyze_time_series(time_series_data, stock_code)

    print("Time series analysis completed. Check the output folder for generated plots.")