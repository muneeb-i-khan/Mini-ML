import numpy as np
import linear_regression as lr

x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 7, 9, 11, 13])

(w,b) = lr.linear_regression(x, y)

xi = float(input("Enter an input value for which you wish to predict the output: "))
yp = w * xi + b
print("Predicted output:", yp)

