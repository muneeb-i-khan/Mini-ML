import numpy as np
import multiple_linear_regression as mlr

x = np.array([[1, 2, 3, 4, 5], [11, 31, 5, 7, 9]])
y = np.array([4, 18])

w, b = mlr.multiple_linear_regression(x, y)

xi = np.array([1,2,6,7,5])
yp = np.dot(w, xi.T) + b
print("Predicted output:", yp)
