from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

iris = load_iris()
iris_data =iris.data
print(iris_data)

iris_label = iris.target
iris_target_name = iris.target_names
print(iris_label)
print(iris_target_name)

import pandas as pd
iris_df = pd.DataFrame(data=iris_data,columns=iris.feature_names)
iris_df['label'] = iris_label
print(iris_df.head(3))

X_train,X_test,y_train,y_test = train_test_split(iris_data,iris_label,test_size=0.2,random_state=11)

dt_clf = DecisionTreeClassifier(random_state=11)
dt_clf.fit(X_train,y_train)

pred = dt_clf.predict(X_test)
print(pred)
print(y_test)
print(pred==y_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,pred))