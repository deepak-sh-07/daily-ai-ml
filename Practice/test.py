from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score


import pandas as pd
df = pd.read_csv('house_price.csv')

sc = StandardScaler()
X = df[["Size_sqft","Bedrooms","Age_years","Distance_to_city_km"]]
y = df["Price"]
# print(X.head())
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled,y_train)
pred = model.predict(X_test_scaled)
# print(pred)
print("MAE:", mean_absolute_error(y_test, pred))
print("R2 Score:", r2_score(y_test, pred)*100)