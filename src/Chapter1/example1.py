import pandas as pd
import os

dataset_path = "/home/tak/github/dataset"

titanic_df = pd.read_csv(os.path.join(dataset_path,'titanic_train.csv'))

print(titanic_df.head())
print("DataFrame 크기 : ",titanic_df.shape)
print(titanic_df.info())
print(titanic_df.describe())

value_counts = titanic_df['Pclass'].value_counts()
print(value_counts)

titanic_pclass = titanic_df['Pclass']
print(type(titanic_pclass))

print(titanic_df.head())
