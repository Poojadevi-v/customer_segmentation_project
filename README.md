import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv(r"C:\Users\pooja\Downloads\thiranex_project\Mall_Customers.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Remove duplicate rows if any
df.drop_duplicates(inplace=True)

print("\nStatistical Summary:")
print(df.describe())

plt.figure(figsize=(6,4))
plt.hist(df['Age'], bins=10)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(6,4))
plt.hist(df['Annual Income (k$)'], bins=10)
plt.title("Annual Income Distribution")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(6,4))
plt.hist(df['Spending Score (1-100)'], bins=10)
plt.title("Spending Score Distribution")
plt.xlabel("Spending Score")
plt.ylabel("Count")
plt.show()

X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(7,5))
plt.plot(range(1,11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.grid(True)
plt.show()

kmeans = KMeans(n_clusters=5, random_state=42)

df['Cluster'] = kmeans.fit_predict(X_scaled)

print("\nDataset with Cluster Labels:")
print(df.head())
plt.figure(figsize=(8,6))

plt.scatter(
    df['Annual Income (k$)'],
    df['Spending Score (1-100)'],
    c=df['Cluster'],
    cmap='viridis',
    s=80
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segmentation using K-Means")
plt.show()

plt.figure(figsize=(6,4))
df['Cluster'].value_counts().sort_index().plot(kind='bar')
plt.title("Customers in Each Cluster")
plt.xlabel("Cluster")
plt.ylabel("Number of Customers")
plt.show()

print("\nCluster Summary:")

summary = df.groupby('Cluster')[['Age',
                                 'Annual Income (k$)',
                                 'Spending Score (1-100)']].mean()

print(summary)

df.to_csv("Customer_Segments.csv", index=False)

print("\nCustomer_Segments.csv has been saved successfully.")

print("\nBusiness Recommendations")

print("""
Cluster 0:
High-income customers with high spending.
Recommendation:
Offer VIP memberships and premium products.

Cluster 1:
High-income customers with low spending.
Recommendation:
Provide personalized offers and discounts.

Cluster 2:
Low-income customers with high spending.
Recommendation:
Introduce cashback and reward programs.

Cluster 3:
Low-income customers with low spending.
Recommendation:
Target them with affordable products.

Cluster 4:
Average income and average spending.
Recommendation:
Maintain engagement through loyalty programs.
""")

print("Project Completed Successfully!")
