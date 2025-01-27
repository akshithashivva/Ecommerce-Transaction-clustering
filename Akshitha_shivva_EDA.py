import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
customers = pd.read_csv("Customers.csv", parse_dates=["SignupDate"])
products = pd.read_csv("Products.csv")
transactions = pd.read_csv("Transactions.csv", parse_dates=["TransactionDate"])

# Merge datasets for analysis
merged_df = pd.merge(transactions, customers, on="CustomerID")
merged_df = pd.merge(merged_df, products, on="ProductID")

# EDA and Visualizations

# 1. Customer Analysis
print("Customer Analysis:")
print(f"Number of Customers: {len(customers)}")
print(f"Customers per Region:\n{customers['Region'].value_counts()}")

plt.figure(figsize=(8, 5))
sns.countplot(x="Region", data=customers)
plt.title("Customer Distribution by Region")
plt.show()

plt.figure(figsize=(10, 5))
sns.histplot(customers["SignupDate"], bins=12, kde=True)
plt.title("Customer Signup Date Distribution")
plt.xlabel("Signup Date")
plt.ylabel("Number of Customers")
plt.show()

# 2. Product Analysis
print("\nProduct Analysis:")
print(f"Number of Products: {len(products)}")
print(f"Products per Category:\n{products['Category'].value_counts()}")

plt.figure(figsize=(8, 5))
sns.countplot(x="Category", data=products)
plt.title("Product Distribution by Category")
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(products["Price"], bins=20, kde=True)
plt.title("Product Price Distribution")
plt.xlabel("Price")
plt.ylabel("Number of Products")
plt.show()

# 3. Transaction Analysis
print("\nTransaction Analysis:")
print(f"Total Transactions: {len(transactions)}")

plt.figure(figsize=(8, 5))
sns.histplot(transactions["Quantity"], bins=10)
plt.title("Transaction Quantity Distribution")
plt.xlabel("Quantity")
plt.ylabel("Number of Transactions")
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(transactions["TotalValue"], bins=20)
plt.title("Transaction Value Distribution")
plt.xlabel("Total Value")
plt.ylabel("Number of Transactions")
plt.show()

plt.figure(figsize=(10, 5))
sns.histplot(merged_df["TransactionDate"], bins=30)
plt.title("Transaction Date Distribution")
plt.xlabel("Transaction Date")
plt.ylabel("Number of Transactions")
plt.show()

# 4. Bivariate Analysis
print("\nBivariate Analysis:")

# Customer Region vs. Product Category
plt.figure(figsize=(10, 6))
sns.countplot(x="Region", hue="Category", data=merged_df)
plt.title("Customer Region vs. Product Category")
plt.xticks(rotation=45)
plt.show()

# Customer Signup Date vs. Average Transaction Value
customer_avg_value = merged_df.groupby("CustomerID").agg({"TotalValue": "mean"}).reset_index()
customer_avg_value = pd.merge(customer_avg_value, customers, on="CustomerID")

plt.figure(figsize=(10, 6))
sns.scatterplot(x="SignupDate", y="TotalValue", data=customer_avg_value)
plt.title("Customer Signup Date vs. Average Transaction Value")
plt.show()

# Product Price vs. Sales Volume
product_sales = merged_df.groupby("ProductID").agg({"Quantity": "sum"}).reset_index()
product_sales = pd.merge(product_sales, products, on="ProductID")

plt.figure(figsize=(10, 6))
sns.scatterplot(x="Price", y="Quantity", data=product_sales)
plt.title("Product Price vs. Sales Volume")
plt.show()

# Customer Purchase History
customer_purchase_freq = merged_df.groupby("CustomerID").agg({"TransactionID": "count"}).reset_index()
customer_purchase_freq.columns = ["CustomerID", "PurchaseFrequency"]

customer_recency = merged_df.groupby("CustomerID").agg({"TransactionDate": "max"}).reset_index()
customer_recency.columns = ["CustomerID", "LastPurchaseDate"]

customer_history = pd.merge(customer_avg_value, customer_purchase_freq, on="CustomerID")
customer_history = pd.merge(customer_history, customer_recency, on="CustomerID")

# Calculate Customer Lifetime Value (CLTV) (Simplified Example)
customer_history["CLTV"] = customer_history["TotalValue"] * customer_history["PurchaseFrequency"]

# Analyze CLTV
print("\nCustomer Lifetime Value (CLTV) Analysis:")
print(customer_history.sort_values(by="CLTV", ascending=False).head())
