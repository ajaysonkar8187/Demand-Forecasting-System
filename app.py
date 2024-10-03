import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
from statsmodels.tsa.arima.model import ARIMA

# Load and preprocess functions
@st.cache_data
def load_data():
    # Load the uploaded datasets
    transactions_1 = pd.read_csv('Transactional_data_retail_01.csv')
    transactions_2 = pd.read_csv('Transactional_data_retail_02.csv')
    customer_demographics = pd.read_csv('CustomerDemographics.csv')
    product_info = pd.read_csv('ProductInfo.csv')
    
    return transactions_1, transactions_2, customer_demographics, product_info

def preprocess_data(transactions_1, transactions_2, customer_demographics, product_info):
    # Combine transactions_1 and transactions_2
    transactions = pd.concat([transactions_1, transactions_2], ignore_index=True)
    
    # Convert 'InvoiceDate' to datetime format
    transactions['InvoiceDate'] = pd.to_datetime(transactions['InvoiceDate'], format='%d-%m-%Y', errors='coerce')
    
    # Filter out canceled orders (negative quantities)
    transactions = transactions[transactions['Quantity'] > 0]
    
    # Merge transactions with customer demographics and product info
    transactions = pd.merge(transactions, customer_demographics, on='Customer ID', how='left')
    transactions = pd.merge(transactions, product_info, on='StockCode', how='left')
    
    return transactions, customer_demographics, product_info

def load_model(stock_code):
    model_path = f'models/arima_model_{stock_code}.joblib'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        return None

def get_forecast(model, steps):
    return model.forecast(steps=steps)

def prepare_time_series(transactions, stock_code):
    # Filter for the selected product
    product_data = transactions[transactions['StockCode'] == stock_code]
    
    # Check if product_data is empty
    if product_data.empty:
        raise ValueError(f"No data found for stock code: {stock_code}")
    
    # Resample to weekly data
    weekly_data = product_data.set_index('InvoiceDate').resample('W')['Quantity'].sum().reset_index()
    
    return weekly_data

def train_and_save_model(transactions, stock_code):
    # Prepare time series data
    weekly_data = prepare_time_series(transactions, stock_code)
    
    # Fit ARIMA model
    model = ARIMA(weekly_data['Quantity'], order=(1,1,1))  # You might need to adjust these parameters
    results = model.fit()

    # Save the model
    if not os.path.exists('models'):
        os.makedirs('models')
    model_path = f'models/arima_model_{stock_code}.joblib'
    joblib.dump(results, model_path)
    st.success(f"Model for stock code {stock_code} trained and saved successfully.")
    return results

# Streamlit App
st.title('Demand Forecasting System')

# Load data
transactions_1, transactions_2, customer_demographics, product_info = load_data()
transactions, customer_demographics, product_info = preprocess_data(transactions_1, transactions_2, customer_demographics, product_info)

# Get top 10 products based on quantity sold
top_10_quantity = transactions.groupby('StockCode')['Quantity'].sum().nlargest(10)

# Display top 10 products
st.subheader('Top 10 Products')
st.table(top_10_quantity)

# User input
selected_stock = st.selectbox('Select a stock code:', top_10_quantity.index)
forecast_weeks = st.slider('Number of weeks to forecast:', 1, 15, 15)

if st.button('Generate Forecast'):
    try:
        # Get historical data
        historical_data = prepare_time_series(transactions, selected_stock)
        
        # Load or train model
        model = load_model(selected_stock)
        if model is None:
            st.info(f"Training new model for stock code: {selected_stock}...")
            model = train_and_save_model(transactions, selected_stock)
        
        # Generate forecast
        forecast = get_forecast(model, forecast_weeks)
        
        # Plot historical data and forecast
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(historical_data['InvoiceDate'], historical_data['Quantity'], label='Historical')
        ax.plot(pd.date_range(start=historical_data['InvoiceDate'].iloc[-1], periods=forecast_weeks + 1, freq='W')[1:], forecast, label='Forecast')
        ax.set_title(f'Demand Forecast for Stock Code {selected_stock}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Quantity')
        ax.legend()
        st.pyplot(fig)
        
        # Display forecast data
        st.subheader('Forecast Data')
        forecast_df = pd.DataFrame({'Date': pd.date_range(start=historical_data['InvoiceDate'].iloc[-1], periods=forecast_weeks + 1, freq='W')[1:], 'Forecast': forecast})
        st.dataframe(forecast_df)
        
        # Option to download forecast as CSV
        csv = forecast_df.to_csv(index=False)
        st.download_button(
            label="Download forecast as CSV",
            data=csv,
            file_name=f'forecast_{selected_stock}.csv',
            mime='text/csv',
        )

        # Error histogram
        historical_forecast = get_forecast(model, len(historical_data))
        errors = historical_data['Quantity'].values - historical_forecast[:len(historical_data)]
        
        st.subheader('Error Distribution')
        fig, ax = plt.subplots()
        ax.hist(errors, bins=20)
        ax.set_title('Error Distribution')
        ax.set_xlabel('Error')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)

    except ValueError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")