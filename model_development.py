import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from data_preprocessing import load_data, preprocess_data, get_top_products
import joblib
import matplotlib.pyplot as plt

def prepare_time_series(transactions, stock_code):
    product_data = transactions[transactions['StockCode'] == stock_code]
    weekly_data = product_data.set_index('TransactionDate').resample('W')['Quantity'].sum().reset_index()
    return weekly_data

def fit_arima(data, order):
    model = ARIMA(data, order=order)
    results = model.fit()
    return results

def fit_ets(data):
    model = ExponentialSmoothing(data, seasonal_periods=52, trend='add', seasonal='add')
    results = model.fit()
    return results

def fit_prophet(data):
    df = data.rename(columns={'TransactionDate': 'ds', 'Quantity': 'y'})
    model = Prophet()
    model.fit(df)
    return model

def forecast_arima(model, steps):
    forecast = model.forecast(steps=steps)
    return forecast

def forecast_ets(model, steps):
    forecast = model.forecast(steps=steps)
    return forecast

def forecast_prophet(model, steps, last_date):
    future_dates = pd.date_range(start=last_date, periods=steps+1, freq='W')[1:]
    future = pd.DataFrame({'ds': future_dates})
    forecast = model.predict(future)
    return forecast['yhat']

def train_and_evaluate_model(data, stock_code):
    tscv = TimeSeriesSplit(n_splits=5)
    
    models = {
        'ARIMA': (fit_arima, forecast_arima, {'order': (1,1,1)}),
        'ETS': (fit_ets, forecast_ets, {}),
        'Prophet': (fit_prophet, forecast_prophet, {})
    }
    
    results = {}
    
    for model_name, (fit_func, forecast_func, params) in models.items():
        mse_scores = []
        mae_scores = []
        
        for train_index, test_index in tscv.split(data):
            train, test = data.iloc[train_index], data.iloc[test_index]
            
            if model_name == 'Prophet':
                model = fit_func(train, **params)
                forecast = forecast_func(model, len(test), train['TransactionDate'].iloc[-1])
            else:
                model = fit_func(train['Quantity'], **params)
                forecast = forecast_func(model, len(test))
            
            mse = mean_squared_error(test['Quantity'], forecast)
            mae = mean_absolute_error(test['Quantity'], forecast)
            mse_scores.append(mse)
            mae_scores.append(mae)
        
        results[model_name] = {
            'RMSE': np.sqrt(np.mean(mse_scores)),
            'MAE': np.mean(mae_scores)
        }
        
        print(f"{model_name} - Stock Code {stock_code}:")
        print(f"RMSE: {results[model_name]['RMSE']}")
        print(f"MAE: {results[model_name]['MAE']}")
        print()
    
    best_model = min(results, key=lambda x: results[x]['RMSE'])
    return models[best_model][0](data['Quantity'], **models[best_model][2]), best_model, results

def save_model(model, stock_code, model_name):
    joblib.dump(model, f'../models/{model_name}_model_{stock_code}.joblib')

def plot_acf_pacf(data, stock_code):
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    plot_acf(data['Quantity'], ax=ax1)
    ax1.set_title(f'ACF for Stock Code {stock_code}')
    plot_pacf(data['Quantity'], ax=ax2)
    ax2.set_title(f'PACF for Stock Code {stock_code}')
    plt.tight_layout()
    plt.savefig(f'../output/acf_pacf_{stock_code}.png')
    plt.close()

if __name__ == "__main__":
    transactions_1, transactions_2, customer_demographics, product_info = load_data()
    transactions, customer_demographics, product_info = preprocess_data(transactions_1, transactions_2, customer_demographics, product_info)
    top_10_quantity, _ = get_top_products(transactions)

    overall_results = {}

    for stock_code in top_10_quantity.index:
        time_series_data = prepare_time_series(transactions, stock_code)
        plot_acf_pacf(time_series_data, stock_code)
        model, best_model_name, results = train_and_evaluate_model(time_series_data, stock_code)
        save_model(model, stock_code, best_model_name)
        overall_results[stock_code] = results

    best_model = min(overall_results, key=lambda x: min(overall_results[x].values(), key=lambda y: y['RMSE'])['RMSE'])
    print(f"\nBest performing model overall: Stock Code {best_model}")
    print(f"Model: {min(overall_results[best_model], key=lambda x: overall_results[best_model][x]['RMSE'])}")
    print(f"RMSE: {min(overall_results[best_model].values(), key=lambda x: x['RMSE'])['RMSE']}")

    print("\nAll models have been trained and saved in the 'models' directory.")