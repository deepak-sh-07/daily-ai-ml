import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder,StandardScaler,minmax_scale

df = pd.read_csv('Drug.csv')
oe = OrdinalEncoder()
le = LabelEncoder()
sc = StandardScaler()
df['Na_to_K'] = minmax_scale(df['Na_to_K'])
df[['Sex','BP','Cholesterol']] = oe.fit_transform(df[['Sex','BP','Cholesterol']])
df['Drug'] = le.fit_transform(df['Drug'])
df['Na_to_K'] = sc.fit_transform(df[['Na_to_K']])
print(df.head())