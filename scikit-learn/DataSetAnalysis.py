from sklearn import datasets
import numpy as np

from sklearn.model_selection import train_test_split
iris = datasets.load_iris()
#split it in features and labels
x = iris.data
y = iris.target


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

print(x_train.shape)
print(x_test.shape)
print(y_test.shape)
print(y_train.shape)