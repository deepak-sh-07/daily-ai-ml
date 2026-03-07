import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
data = datasets.load_wine()
X = data.data
y = data.target

df = pd.DataFrame(X, columns=data.feature_names) #changing to dataframe to get the colums values easily same we do when we import datasets.
df['Wine class'] = y
#print(df)

#preprocessing
#print(df.isna().sum())
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

#KMeans
from sklearn.cluster import KMeans
wss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init = 'k-means++', random_state=0)
    kmeans.fit(X)
    wss.append(kmeans.inertia_)

f3,ax = plt.subplots(figsize=(10,5))
plt.plot(range(1,11),wss)
plt.title('the elbow method ')
plt.xlabel('Number of clusters ')
plt.ylabel('Wss')
#plt.show() #elbow at 3


N = 3
k_means = KMeans(n_clusters=3,
    init='k-means++',
    n_init=20,
    max_iter=500,
    random_state=42)
k_means.fit(X)
labels = k_means.labels_
# print(labels)

from sklearn.metrics import adjusted_rand_score,accuracy_score
print(adjusted_rand_score(y, labels))



