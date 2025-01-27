import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load the datasets (assuming you have the necessary CSV files)
customers = pd.read_csv("Customers.csv", parse_dates=["SignupDate"])
products = pd.read_csv("Products.csv")
transactions = pd.read_csv("Transactions.csv", parse_dates=["TransactionDate"])

# Merge datasets
merged_df = pd.merge(transactions, customers, on="CustomerID")
merged_df = pd.merge(merged_df, products, on="ProductID")

# Create customer-product interaction matrix
customer_product_matrix = merged_df.pivot_table(index='CustomerID', columns='ProductID', values='Quantity', fill_value=0)

def find_lookalikes(customer_id, customer_product_matrix, n_lookalikes=3):


  try:
      customer_vector = customer_product_matrix.loc[customer_id].values.reshape(1, -1)
      similarities = cosine_similarity(customer_vector, customer_product_matrix)
      similar_customers = similarities[0].argsort()[::-1][1:n_lookalikes + 1]  # Exclude the customer itself
      lookalikes = []
      for i, customer in enumerate(similar_customers):
          lookalikes.append((customer_product_matrix.index[customer], similarities[0][customer]))
      return lookalikes
  except KeyError:
      print(f"Customer ID {customer_id} not found in data. Skipping...")
      return []  # Return an empty list if customer ID is missing

# Example usage:
customer_id = "C0001"  # Replace with the actual customer ID
lookalike_customers = find_lookalikes(customer_id, customer_product_matrix)

if lookalike_customers:
    print(f"Lookalikes for customer {customer_id}:")
    for customer, score in lookalike_customers:
        print(f"Customer ID: {customer}, Similarity Score: {score}")
else:
    print(f"No lookalikes found for customer {customer_id}.")
