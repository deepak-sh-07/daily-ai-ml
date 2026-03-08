import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Mall_Customers.csv')
print(df)
print(df.isna().sum())
print(df.info())
print(df.describe())
print(df[['Age','Annual Income (k$)','Spending Score (1-100)']].mean())


#Age vs Income
#Age vs Spending Score
#Income vs Spending Score
sns.scatterplot(x='Age',y='Annual Income (k$)',data=df)
sns.scatterplot(x='Age',y='Spending Score (1-100)',data=df)
sns.scatterplot(x='Annual Income (k$)',y='Spending Score (1-100)',data=df)
plt.show()

#five behavioural clusters
#Low income – low spending → Budget customers
#Low income – high spending → Impulsive/enthusiastic shoppers
#Middle income – medium spending → Average/standard customers
#High income – low spending → Conservative wealthy customers
#High income – high spending → Premium/VIP customers

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values
scaler = StandardScaler()  
X_scaled = scaler.fit_transform(X)

#elbow method to find the optimal number of clusters
wss =[]
for i in range (1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',random_state=0)
    kmeans.fit(X_scaled)
    wss.append(kmeans.inertia_)
f3,ax = plt.subplots(figsize=(10,5))
plt.plot(range(1,11),wss)
plt.title('the elbow method ')
plt.xlabel('Number of clusters ')
plt.ylabel('Wss')
plt.show() #elbow at 5

N = 5
k_means = KMeans(n_clusters=N,
    init='k-means++',
    n_init=20,
    max_iter=500,
    random_state=42)
k_means.fit(X_scaled)
labels = k_means.labels_
#print(labels)

df['Cluster'] = labels
#print(df.head())
plt.figure(figsize=(10,6))
centroids = scaler.inverse_transform(k_means.cluster_centers_)

sns.scatterplot(
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    hue='Cluster',
    palette='tab10',
    data=df
)

plt.scatter(
    centroids[:,0],
    centroids[:,1],
    s=300,
    c='black',
    marker='X',
    label='Centroids'
)

plt.title("Customer Segments with Centroids")
plt.legend()
plt.show()
