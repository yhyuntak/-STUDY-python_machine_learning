
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

titanic_drop_df = titanic_df.drop('Age_0',axis=1)
print(titanic_drop_df.head(3))

drop_result = titanic_df.drop(['Age_0','Age_by_10','Family_No'],axis=1,inplace=True)
print(drop_result)
print(titanic_df.head(3))

titanic_df.drop([0,1,2],axis=0,inplace=True)
print(titanic_df.head(3))