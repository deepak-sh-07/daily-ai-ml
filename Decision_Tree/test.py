import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
df = pd.read_csv('Drug.csv')

#Preprocessing
#print(df.head())
#print(df.isna().sum())
#print(df[df.duplicated()])
#print(df.info())

# x = df['Sex'].value_counts()
# print(x)
# sns.countplot(data=df,x='Sex')
# plt.show()

oe = OrdinalEncoder()
df['BP'] = oe.fit_transform(df[['BP']])
df['Cholesterol'] = oe.fit_transform(df[['Cholesterol']])
df['Drug'] = oe.fit_transform(df[['Drug']])
df['Sex'] = oe.fit_transform(df[['Sex']])
#print(df)

X = df.iloc[:,0:-1]
y = df.iloc[:, -1]
# print(X)
# print(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
pred = model.predict(X_test)
#print(pred)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, pred)
print(acc)