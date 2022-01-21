
## page 49
import pandas as pd
import numpy as np
col_name1 = ['col1']
list1 = [1,2,3]
array1 = np.array(list1)
print('array1 shape : ',array1.shape)
df_list1 = pd.DataFrame(list1,columns=col_name1)
print(df_list1)
df_array1 = pd.DataFrame(array1,columns=col_name1)
print(df_array1)

col_name2 = ['col1','col2','col3']
list2 = [[1,2,3],[11,12,13]]
array2 = np.array(list2)
df_list2 = pd.DataFrame(list2,columns=col_name2)
df_array2 = pd.DataFrame(array2,columns=col_name2)
print(df_list2)
print(df_array2)

dict = {'col1':[1,11],'col2':[2,22]}
df_dict = pd.DataFrame(dict)
print(df_dict)

array3 = df_dict.values
print(array3.shape)
print(array3)

list3 = df_dict.values.tolist()
dict3 = df_dict.to_dict('list')
print(list3)
print(dict3)


