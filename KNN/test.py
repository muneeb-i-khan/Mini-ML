import knn as KNN
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

colors = ListedColormap(['red', 'blue', 'green'])

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

iris = datasets.load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1234)

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=colors, edgecolor='k', s=20)
plt.show()

clf = KNN.KNN(k=5)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print("KNN classification accuracy", accuracy(y_test, predictions))