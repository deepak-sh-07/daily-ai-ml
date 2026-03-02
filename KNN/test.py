import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import math
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

data = datasets.load_wine(as_frame=True) #to import the wine dataset and as a dataframe

#print(data)
X = data.data 
y = data.target 
names = data.target_names
# print(names)
df = pd.DataFrame(X, columns=data.feature_names)
df['wine class'] = data.target
df['wine class'] = df['wine class'].replace({0: 'class_0', 1: 'class_1', 2: 'class_2'}) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) #to split the data into training and testing sets

math.sqrt(len(y_test)) 
knn = KNeighborsClassifier(n_neighbors=5) 
knn.fit(X_train, y_train) 

predictions = knn.predict(X_test)
# print(predictions)

print(metrics.accuracy_score(y_test, predictions))

sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)
predictions_scaled = knn_scaled.predict(X_test_scaled)
print(metrics.accuracy_score(y_test, predictions_scaled))