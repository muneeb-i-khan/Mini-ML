import numpy as np

def z_score_normalization(x):
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    non_zero_std = std != 0.0
    x_normalized = np.zeros_like(x)
    x_normalized[:, non_zero_std] = (x[:, non_zero_std] - mean[non_zero_std]) / std[non_zero_std]

    return x_normalized

def multiple_linear_regression(x, y, alpha=0.01, max_iterations=1000):
    x_normalized = z_score_normalization(x)
    xLen = x_normalized.shape[1]
    m = y.shape[0]
    w = np.zeros(xLen)
    for i in range(xLen):
        w[i] = float(input("Enter the value of w[" + str(i) + "]: "))
    b = float(input("Enter the value of b: "))

    if alpha > 0.1:
        print("Warning: Learning rate is too high. Consider reducing it for a stable convergence.")

    try:
        w, b = gradient_descent(w, b, m, x_normalized, y, alpha, max_iterations)
    except OverflowError:
        print("Overflow error occurred. Try reducing the learning rate or normalizing the input features.")
        return

    print("Optimized w:", w)
    print("Optimized b:", b)
    print("Cost:", calculate_cost(x_normalized, y, m, w, b))
    return w, b

def calculate_cost(x, y, m, w, b):
    yp = np.dot(x, w) + b
    cost = np.sum((yp - y) ** 2)
    cost = cost / (2 * m)
    return cost

def gradient_descent(w, b, m, x, y, alpha, max_iterations):
    iteration = 0
    while iteration < max_iterations:
        cost = calculate_cost(x, y, m, w, b)
        temp_w = w - alpha * calculate_derivative_w(x, y, m, w, b)
        temp_b = b - alpha * calculate_derivative_b(x, y, m, w, b)

        if np.all(np.abs(temp_w - w) < 1e-9) and np.abs(temp_b - b) < 1e-9:
            break

        w = temp_w
        b = temp_b
        iteration += 1

    return w, b

def calculate_derivative_w(x, y, m, w, b):
    yp = np.dot(x, w) + b
    derivative_w = np.dot((yp - y), x)
    derivative_w = derivative_w / m
    return derivative_w

def calculate_derivative_b(x, y, m, w, b):
    yp = np.dot(x, w) + b
    derivative_b = np.sum((yp - y))
    derivative_b = derivative_b / m
    return derivative_b
