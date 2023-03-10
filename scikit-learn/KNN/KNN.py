# use odd number for k -> advised
# if big data set -> high number of k
# if small data set -> low number of k
# weights -> a "uniform" parameter gives equal importance to all datapoint
# weights -> a "distant" parameter gives greater importance to closer datapoint

import numpy as np
import pandas as pd
from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('car.data')

x = data[[
    'buying',
    'maint',
    'saftey'
]].values
y = data[['class']]
# converting the data for x

Le = LabelEncoder()
for i in range(len(x[0])):
    x[:, i] = Le.fit_transform(x[:, i])

# converting the data for y

label_mapping = {
    'unacc': 0,
    'acc': 1,
    'good': 2,
    'vgood': 3
}

y['class'] = y['class'].map(label_mapping)
y = np.array(y)

#CREATING A MODEL

knn = neighbors.KNeighborsClassifier(n_neighbors=25, weights='uniform')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

knn.fit(x_train, y_train)

prediction = knn.predict(x_test)

accuracy = metrics.accuracy_score(y_test, prediction);
print("Predictions: ", prediction)
print("Accuracy: ", accuracy)

a = 56
print("Actual Value: ", y[a])
print("Predicted Value: ", knn.predict(x)[a])
