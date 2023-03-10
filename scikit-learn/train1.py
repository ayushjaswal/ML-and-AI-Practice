from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
#split it in features and labels
x = iris.data
y = iris.target

print(x.shape)
print(y.shape)
