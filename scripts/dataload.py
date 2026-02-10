import pandas as pd
df=pd.read_csv('data/raw/orders.csv')
print("Dataset loaded successfully")
print(df.head())
print(df.shape)
print(df.info())
print(df.describe())
print(df.columns)
print(df.isnull().sum())
