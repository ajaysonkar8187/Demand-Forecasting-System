import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    transactions_1 = pd.read_csv('../data/Transactional_data_retail_01.csv')
    transactions_2 = pd.read_csv('../data/Transactional_data_retail_02.csv')
    customer_demographics = pd.read_csv('../data/CustomerDemographics.csv')
    product_info = pd.read_csv('../data/ProductInfo.csv')
    
    return transactions_1, transactions_2, customer_demographics, product_info

def preprocess_data(transactions_1, transactions_2, customer_demographics, product_info):
    # Combine transaction data
    transactions = pd.concat([transactions_1, transactions_2], ignore_index=True)
    
    # Convert date column to datetime
    transactions['TransactionDate'] = pd.to_datetime(transactions['TransactionDate'])
    
    # Merge with product info to get prices
    transactions = transactions.merge(product_info[['StockCode', 'Price']], on='StockCode', how='left')
    
    # Calculate revenue
    transactions['Revenue'] = transactions['Quantity'] * transactions['Price']
    
    return transactions, customer_demographics, product_info

def get_top_products(transactions, n=10):
    # Top N products by quantity sold
    top_n_quantity = transactions.groupby('StockCode')['Quantity'].sum().sort_values(ascending=False).head(n)
    
    # Top N products by revenue
    top_n_revenue = transactions.groupby('StockCode')['Revenue'].sum().sort_values(ascending=False).head(n)
    
    return top_n_quantity, top_n_revenue

def customer_level_summary(transactions):
    return transactions.groupby('CustomerID').agg({
        'TransactionDate': 'count',
        'Quantity': 'sum',
        'Revenue': 'sum'
    }).rename(columns={'TransactionDate': 'TransactionCount'})

def product_level_summary(transactions):
    return transactions.groupby('StockCode').agg({
        'TransactionDate': 'count',
        'Quantity': 'sum',
        'Revenue': 'sum'
    }).rename(columns={'TransactionDate': 'TransactionCount'})

def transaction_level_summary(transactions):
    return transactions.agg({
        'Quantity': ['sum', 'mean', 'median', 'std'],
        'Revenue': ['sum', 'mean', 'median', 'std']
    })

def plot_top_products(top_10_quantity, top_10_revenue):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    top_10_quantity.plot(kind='bar', ax=ax1)
    ax1.set_title('Top 10 Products by Quantity Sold')
    ax1.set_xlabel('Stock Code')
    ax1.set_ylabel('Quantity')
    
    top_10_revenue.plot(kind='bar', ax=ax2)
    ax2.set_title('Top 10 Products by Revenue')
    ax2.set_xlabel('Stock Code')
    ax2.set_ylabel('Revenue')
    
    plt.tight_layout()
    plt.savefig('../output/top_products.png')
    plt.close()

if __name__ == "__main__":
    transactions_1, transactions_2, customer_demographics, product_info = load_data()
    transactions, customer_demographics, product_info = preprocess_data(transactions_1, transactions_2, customer_demographics, product_info)
    top_10_quantity, top_10_revenue = get_top_products(transactions)
    
    customer_summary = customer_level_summary(transactions)
    product_summary = product_level_summary(transactions)
    transaction_summary = transaction_level_summary(transactions)
    
    plot_top_products(top_10_quantity, top_10_revenue)
    
    print("Data preprocessing completed.")
    print(f"Total transactions: {len(transactions)}")
    print("\nCustomer-level summary:")
    print(customer_summary.describe())
    print("\nProduct-level summary:")
    print(product_summary.describe())
    print("\nTransaction-level summary:")
    print(transaction_summary)
    
    # Save summaries to CSV
    customer_summary.to_csv('../output/customer_summary.csv')
    product_summary.to_csv('../output/product_summary.csv')
    transaction_summary.to_csv('../output/transaction_summary.csv')