# imporing libraries
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


iris = datasets.load_iris()

# 0 - Iris-Setosa
# 1 - Iris-Versicolour
# 2 - Iris-Virginica

features = iris.data
labels = iris.target

#Splitting the Data into training and testing sets

features_train, features_test, labels_trains, labels_test = train_test_split(features, labels, test_size=0.2)

# Training the classifier

clf = KNeighborsClassifier()
clf.fit(features_train, labels_trains)

prdc = clf.predict(features_test)[10]

if prdc == 0:
    print("Setosa")
elif prdc == 1:
    print("Versicolour")
else:
    print("Virginica")

