import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Credit Card Customer Data.csv')
df = df.drop(['Customer Key'], axis=1)
df = df.drop(['Sl_No'], axis=1)
# print(df.head())
# print(df.info())
# print(df.agg(['min', 'max', 'mean']))

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df_scaled = pd.DataFrame(
scaler.fit_transform(df),
columns=df.columns
)
# print(df_scaled.head())

from sklearn.cluster import KMeans
wss=[]
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',random_state=42)
    kmeans.fit(df_scaled)
    wss.append(kmeans.inertia_)
# plt.plot(range(1,11),wss)
# plt.title('Elbow Method')
# plt.xlabel('Number of clusters')
# plt.ylabel('WSS')
# plt.show()

kmeans = KMeans(n_clusters=3,init='k-means++',random_state=42)
kmeans.fit(df_scaled)
labels = kmeans.labels_
df['Cluster'] = labels

# print(df.head())
#Avg_Credit_Limit  Total_Credit_Cards  Total_visits_bank  Total_visits_online  Total_calls_made  Cluster
# plt.scatter(df['Avg_Credit_Limit'], df['Total_Credit_Cards'], c=df['Cluster'], cmap='viridis')
# plt.xlabel('Avg_Credit_Limit')
# plt.ylabel('Total_Credit_Cards')
# plt.title('Customer Segmentation')
# plt.show()

cluster_summary = df.groupby('Cluster').mean()

# print(cluster_summary)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca_components = pca.fit_transform(df_scaled)
pca_df = pd.DataFrame(pca_components, columns=['PCA1','PCA2'])
pca_df['Cluster'] = df['Cluster']
plt.scatter(pca_df['PCA1'], pca_df['PCA2'], c=pca_df['Cluster'], cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Customer Segmentation (PCA Projection)')
plt.show()
