import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('./creditcard.csv')[:80000]
df.head(3)
X = df.drop(columns=['Time','Amount','Class']).values
y = df['Class'].values
# print(f"Shape of X: {X.shape}, Shape of y: {y.shape}")
# print("Fraud count:", y.sum())
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X,y)
y_pred = model.predict(X).sum()
# print("Predicted labels:", y_pred)

from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid={'class_weight': [{0:1,1:v}for v in range(1,4)]}, cv=5,n_jobs=-1)
grid.fit(X,y)
print("Best parameters:", grid.best_params_)
print("Best score:", grid.best_score_)  
      
values = pd.DataFrame(grid.cv_results_)
print(values)
