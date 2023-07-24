import numpy as np

def normalize_features(x):
    x_mean = np.mean(x)
    x_std = np.std(x)
    x_normalized = (x - x_mean) / x_std
    return x_normalized

def linear_regression(x, y):
    x_normalized = normalize_features(x)  
    m = len(x_normalized)  

    w = float(input("Enter the value of w: "))
    b = float(input("Enter the value of b: "))
    alpha = float(input("Enter the value of alpha: "))

    if alpha > 0.1:
        print("Warning: Learning rate is too high. Consider reducing it for a stable convergence.")

    try:
        w, b = gradient_descent(w, b, m, x_normalized, y, alpha)  
    except OverflowError:
        print("Overflow error occurred. Try reducing the learning rate or normalizing the input features.")
        return

    print("Optimized w:", w)
    print("Optimized b:", b)
    print("Cost:", calculate_cost(x_normalized, y, m, w, b))  

def calculate_cost(x, y, m, w, b):
    cost = 0
    for i in range(m):
        yp = w * x[i] + b
        cost += (yp - y[i]) ** 2
    cost = cost / (2 * m)
    return cost

def gradient_descent(w, b, m, x, y, alpha):
    while True:
        cost = calculate_cost(x, y, m, w, b)
        temp_w = w - alpha * calculate_derivative_w(x, y, m, w, b)
        temp_b = b - alpha * calculate_derivative_b(x, y, m, w, b)
        if abs(temp_w - w) < 1e-9 and abs(temp_b - b) < 1e-9:
            break
        w = temp_w
        b = temp_b
    return w, b

def calculate_derivative_w(x, y, m, w, b):
    derivative_w = 0
    for i in range(m):
        yp = w * x[i] + b
        derivative_w += (yp - y[i]) * x[i]
    derivative_w = derivative_w / m
    return derivative_w

def calculate_derivative_b(x, y, m, w, b):
    derivative_b = 0
    for i in range(m):
        yp = w * x[i] + b
        derivative_b += (yp - y[i])
    derivative_b = derivative_b / m
    return derivative_b


