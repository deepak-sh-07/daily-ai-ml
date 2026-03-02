import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt
houses = pd.read_csv('house_price.csv')
X = houses[["Size_sqft","Bedrooms","Distance_to_city_km"]]
y = houses["Price"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = LinearRegression()
model.fit(X_train,y_train)
print(model.score(X_test,y_test)*100)
print(model.score(X_train,y_train)*100)
data = pd.DataFrame([
    {
        "Size_sqft":2000,
        "Bedrooms":3,
        "Distance_to_city_km":10
    }
]) 
# print(model.predict(data))
# print(model.coef_)
# print(model.intercept_)

train_sizes, train_scores, val_scores = learning_curve(
    model,
    X,
    y,
    cv=5,
    scoring='r2',
    train_sizes=np.linspace(0.1, 1.0, 10),
    random_state=42
)

train_mean = np.mean(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)

plt.plot(train_sizes, train_mean, label="Training R2")
plt.plot(train_sizes, val_mean, label="Validation R2")

plt.xlabel("Training Set Size")
plt.ylabel("R2 Score")
plt.title("Learning Curve")
plt.legend()
plt.show()
