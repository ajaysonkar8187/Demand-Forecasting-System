import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import load_data, preprocess_data, get_top_products, customer_level_summary, product_level_summary, transaction_level_summary

def plot_sales_over_time(transactions):
    plt.figure(figsize=(12, 6))
    transactions.groupby('TransactionDate')['Quantity'].sum().plot()
    plt.title('Total Sales Over Time')
    plt.xlabel('Date')
    plt.ylabel('Quantity Sold')
    plt.savefig('../output/sales_over_time.png')
    plt.close()

def plot_revenue_over_time(transactions):
    plt.figure(figsize=(12, 6))
    transactions.groupby('TransactionDate')['Revenue'].sum().plot()
    plt.title('Total Revenue Over Time')
    plt.xlabel('Date')
    plt.ylabel('Revenue')
    plt.savefig('../output/revenue_over_time.png')
    plt.close()

def plot_customer_distribution(customer_summary):
    plt.figure(figsize=(12, 6))
    sns.histplot(customer_summary['TransactionCount'], kde=True)
    plt.title('Distribution of Transactions per Customer')
    plt.xlabel('Number of Transactions')
    plt.ylabel('Frequency')
    plt.savefig('../output/customer_transaction_distribution.png')
    plt.close()

def plot_product_distribution(product_summary):
    plt.figure(figsize=(12, 6))
    sns.histplot(product_summary['Quantity'], kde=True)
    plt.title('Distribution of Quantity Sold per Product')
    plt.xlabel('Quantity Sold')
    plt.ylabel('Frequency')
    plt.savefig('../output/product_quantity_distribution.png')
    plt.close()

if __name__ == "__main__":
    transactions_1, transactions_2, customer_demographics, product_info = load_data()
    transactions, customer_demographics, product_info = preprocess_data(transactions_1, transactions_2, customer_demographics, product_info)
    top_10_quantity, top_10_revenue = get_top_products(transactions)
    
    plot_sales_over_time(transactions)
    plot_revenue_over_time(transactions)
    
    customer_summary = customer_level_summary(transactions)
    product_summary = product_level_summary(transactions)
    transaction_summary = transaction_level_summary(transactions)
    
    plot_customer_distribution(customer_summary)
    plot_product_distribution(product_summary)
    
    print("EDA completed. Check the output folder for generated plots.")
    print("\nCustomer-level summary:")
    print(customer_summary.describe())
    print("\nProduct-level summary:")
    print(product_summary.describe())
    print("\nTransaction-level summary:")
    print(transaction_summary)