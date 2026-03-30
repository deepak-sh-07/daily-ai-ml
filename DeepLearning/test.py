import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('Churn_Modelling.csv')
# print(df.head())
X = df.iloc[:, 3:12]
print(X.head())