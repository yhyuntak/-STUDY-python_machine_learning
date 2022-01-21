
import pandas as pd
import numpy as np

titanic_df = pd.read_csv('../../dataset/titanic_train.csv')
titanic_df['Age_0'] = 0
print(titanic_df.head(3))

titanic_df['Age_by_10'] = titanic_df['Age']*10
titanic_df['Family_No'] = titanic_df['SibSp']+titanic_df['Parch']+1
print(titanic_df.head(3))

titanic_df['Age_by_10'] = titanic_df['Age_by_10']+100
print(titanic_df.head(3))