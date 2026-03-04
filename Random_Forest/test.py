import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
df = pd.read_csv('CarData.csv')
#print(df.head())

#PreProcessing
col_names = ['buying','maint','doors','persons','lug_boot','safety','class']
df.columns = col_names 
#print(df.head())
#print(df.info())

#Encoding
oe = OrdinalEncoder()
df['buying'] = oe.fit_transform(df[['buying']])
df['maint'] = oe.fit_transform(df[['maint']])
df['doors'] = oe.fit_transform(df[['doors']])
df['persons'] = oe.fit_transform(df[['persons']])
df['lug_boot'] = oe.fit_transform(df[['lug_boot']])
df['safety'] = oe.fit_transform(df[['safety']])
df['class'] = oe.fit_transform(df[['class']])

X = df.iloc[:,0:-1]
y = df.iloc[:,-1]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

#Model Building
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
pred1 = rfc.predict(X_test)
# print(accuracy_score(pred1,y_test))

from yellowbrick.model_selection import validation_curve
num_est = [100,200,450,500,750,1000]
#print(validation_curve(rfc,X=X_train,y=y_train,param_name='n_estimators',param_range=num_est,cv=3,scoring='accuracy'))

depth_vals = [5,10,15,20,25,30]
#print(validation_curve(rfc,X=X_train,y=y_train,param_name='max_depth',param_range=depth_vals,cv=3,scoring='accuracy'))

min_samp = [2,4,6,8,10]
#print(validation_curve(rfc,X=X_train,y=y_train,param_name='min_samples_split',param_range=min_samp,cv=3,scoring='accuracy'))

rfc2 = RandomForestClassifier(n_estimators=500,max_depth=20,min_samples_split=4)
rfc2.fit(X_train,y_train)
pred2 = rfc2.predict(X_test)
print(accuracy_score(pred2,y_test))

feature_scores = pd.Series(rfc2.feature_importances_,index=X.columns).sort_values(ascending=False)
print(feature_scores)

# sns.barplot(x=feature_scores,y=feature_scores.index)
# plt.xlabel('Feature Importance Score')
# plt.show()

rfc3 = RandomForestClassifier()
Xn = df.drop(['doors','lug_boot','maint'],axis=1)
yn = df['class']
X_train,X_test,y_train,y_test = train_test_split(Xn,yn,test_size=0.3,random_state=0)
rfc3.fit(X_train,y_train)   
pred3 = rfc3.predict(X_test)
print(accuracy_score(pred3,y_test))