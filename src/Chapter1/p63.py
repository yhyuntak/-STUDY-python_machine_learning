import pandas as pd
import numpy as np

titanic_df = pd.read_csv('../../dataset/titanic_train.csv')

print(titanic_df['Pclass'].head(3))
print(titanic_df)
print(titanic_df[['Survived','Pclass']].head(3))

print(titanic_df[0:3])
print(titanic_df[titanic_df['Pclass']==3].head(3))

