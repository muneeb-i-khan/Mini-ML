from numpy import sqrt, sum
import numpy as np
from collections import Counter

class KNN:

    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)
    
    def euclidean_distance(self, x1, x2):
        return sqrt(sum((x1 - x2)**2))
    
    def _predict(self, x):
        distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = self.most_common(k_nearest_labels)
        return most_common[0][0]
    
    def most_common(self, lst):
        data = Counter(lst)
        return data.most_common()
