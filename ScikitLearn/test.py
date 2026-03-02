from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor()  
model.fit(X_train, y_train)

pipe = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=1))
pipe.fit(X_train, y_train)
# print("Test R² without scaling:", model.score(X_test, y_test) * 100)
# print("Test R² with scaling   :", pipe.score(X_test, y_test) * 100)
mod = GridSearchCV( estimator=pipe,
                    param_grid={'kneighborsregressor__n_neighbors': [1, 3, 5, 7, 9]},
                    cv=5)
mod.fit(X_train, y_train) 
pred = pipe.predict(X_test)
plt.scatter(pred, y_test)
plt.show()