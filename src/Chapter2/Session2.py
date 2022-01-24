from sklearn.datasets import load_iris

iris_data = load_iris()
print(iris_data.keys())

print(iris_data.data)
print(iris_data.target)
print(iris_data.target_names)
print(iris_data.feature_names)
