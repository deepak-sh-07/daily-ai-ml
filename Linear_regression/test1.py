import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import StandardScaler


import numpy as np
import matplotlib.pyplot as plt
 
houses = pd.read_csv('house_price.csv')
X = houses[["Size_sqft","Bedrooms","Distance_to_city_km"]]
y = houses["Price"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)  
X_test_scaled = scaler.transform(X_test) 

model = LinearRegression()
model.fit(X_train_scaled,y_train)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)

# Lasso
lasso = Lasso(alpha=1.0)
lasso.fit(X_train_scaled, y_train)
print("Train R²:")
print("LR   :", model.score(X_train_scaled, y_train) * 100)
print("Ridge:", ridge.score(X_train_scaled, y_train) * 100)
print("Lasso:", lasso.score(X_train_scaled, y_train) * 100)

print("\nTest R²:")
print("LR   :", model.score(X_test_scaled, y_test) * 100)
print("Ridge:", ridge.score(X_test_scaled, y_test) * 100)
print("Lasso:", lasso.score(X_test_scaled, y_test) * 100)

print("\nCoefficients:")
print("LR   :", model.coef_)
print("Ridge:", ridge.coef_)
print("Lasso:", lasso.coef_)


       
