import pandas as pd
import numpy as np

titanic_df = pd.read_csv('../../dataset/titanic_train.csv')
indexes = titanic_df.index
print(indexes)
print(indexes.values)

series_fare = titanic_df['Fare']
print(series_fare.max())
print(series_fare.sum())
print(sum(series_fare))
print((series_fare+3).head(3))

titanic_reset_df = titanic_df.reset_index(inplace=False)
print(titanic_reset_df.head(3))

value_counts = titanic_df['Pclass'].value_counts()
print(value_counts)
new_value_counts = value_counts.reset_index(inplace=False)
print(new_value_counts)