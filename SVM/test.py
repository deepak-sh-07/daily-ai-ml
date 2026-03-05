import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
iris = datasets.load_iris()
X = iris.data
y = iris.target
#print(X.shape)

df = pd.DataFrame(X,columns=iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].replace({0:'setosa',1:'versicolor',2:'virginica'})
# print(df)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

svm = SVC(kernel='linear',random_state=0)
svm.fit(X_train,y_train)
y_pred = svm.predict(X_test)
#print("Predicted labels:", y_pred)
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
print("accuracy:",accuracy_score(y_test,y_pred))