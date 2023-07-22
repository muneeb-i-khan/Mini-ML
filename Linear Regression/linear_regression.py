def linear_regression(x,y):
    m = len(x)
    w = float(input("Enter the value of w: "))
    b = float(input("Enter the value of b: "))
    print(calculate_cost(x,y,m,w,b))

def calculate_cost(x,y,m,w,b):
    cost = 0
    for i in range(m):
        yp = w*x[i] + b
        cost += (yp - y[i])**2
    cost = cost/(2*m)
    return cost
