import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
customers_df = pd.read_csv("Customers.csv")

# Load transaction data
transactions_df = pd.read_csv("Transactions (1).csv")
customer_agg = transactions_df.groupby("CustomerID").agg({
    "TransactionID": "count",  # Number of transactions
    "TotalValue": "sum"       # Total spending
}).rename(columns={"TransactionID": "NumTransactions", "TotalValue": "TotalSpending"})

# Join customer-level features with customer profile data
merged_df = customers_df.merge(customer_agg, on="CustomerID", how="left")
merged_df["SignupDays"] = (pd.to_datetime("today") - pd.to_datetime(merged_df["SignupDate"])).dt.days

# 3. Handle Missing Values (if any)
merged_df = merged_df.fillna(0)  # Replace missing values with 0 (e.g., for customers with no transactions)

# 4. Feature Scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(merged_df[['NumTransactions', 'TotalSpending', 'SignupDays']])
db_index = davies_bouldin_score(scaled_data, kmeans.labels_)
print(f"Davies-Bouldin Index: {db_index}")

# --- Visualization ---

# Visualize clusters (example with 2D scatter plot - consider using PCA for higher dimensions)
plt.figure(figsize=(8, 6))
sns.scatterplot(x="NumTransactions", y="TotalSpending", hue="Cluster", data=merged_df)
plt.title(f"Customer Segmentation (K-Means, {n_clusters} Clusters)")
plt.xlabel("Number of Transactions")
plt.ylabel("Total Spending")
plt.show()

# --- Report ---

print(f"Number of Clusters: {n_clusters}")
print(f"Davies-Bouldin Index: {db_index}")
