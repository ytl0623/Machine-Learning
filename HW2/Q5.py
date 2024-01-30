""""
Following 4), perform polynomial regression with degree 14 by varying the number of
training data points m = 10, 80, 320. Show the five-fold cross-validation errors and the
fitting plots. Compare the results to those in 4).
"""

import numpy as np
import math
import matplotlib.pyplot as plt

def generate_sine_data(point):
    epsilon = np.random.normal(0, 0.04, point)
    x = np.linspace(0, 1, point)
    y = np.sin(2 * x * np.pi) + epsilon
    return x, y

def polynomial_regression(x, degree):
    x3 = np.array([])

    for i in range(int(degree)):
        x2 = np.array([])

        if (i):
            x2 = x**(i + 1)
            x3 = np.column_stack((x3, x2))
        else:
            x3 = np.column_stack((x, np.ones(x.shape[0])))
            x3 = np.fliplr(x3)

    x3 = np.fliplr(x3)
    return x3

def poly_compute(x, y):
    wlin = np.matmul(np.matmul (np.linalg.inv(np.matmul(x.T , x)) , x.T) , y)
    return wlin

def training_error(pred, train):
    num = np.prod(pred.shape)
    error = np.linalg.norm(pred - train) ** 2 / num  # 將pred, train相減取2-norm後平方再除以資料數量
    return error

def kfold_split(a, begin = 15, percent = 0):
    arr = np.array([])
    k = math.floor(a.size * percent)
    valid = a[begin : begin + k]

    if (begin):
        arr = a[ : begin ]

    arr2 = a[begin + k :]
    train = np.append(arr, arr2)
    return train, valid

def kfold(x, y, k, degree):
    begin = 0
    terr = 0
    size = x.size

    for _ in range(k):
        x_train,x_test = kfold_split(x, begin, 1/k)
        y_train,y_test = kfold_split(y, begin, 1/k)
        x_2 = polynomial_regression(x, degree)
        wlin = poly_compute(x_2, y)

        pred = wlin[0] * x_test + wlin[1]
        err = training_error(pred, y_test)
        begin += math.floor(size * 1/k)
        terr += err

    return terr / k

def Q5perDegree(x, y, point):
    x_2 = polynomial_regression(x, 14)
    wlin = poly_compute(x_2, y) #!!!

    print("Q5 Data Points %d Polynomial Regression" %point)

    p = np.poly1d(wlin.flatten())
    pred = p(x)

    print("Q5 Polynomial Regression Training Error: " + str(training_error(pred, y)))

    print("Q5 Polynomial Regression Five-Fold Cross Validation Error: " + str(kfold(x, y, 5, 14)) + "\n")  #!!!

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.scatter(x, y, label = "Training Data")
    plt.title("Q5: Polynomial Regression Data Points %d" %point)
    plt.plot(x,  pred, "r-", label = "Fitting Function")
    plt.legend()
    plt.show()

    return

def main():
    x, y = generate_sine_data(10)
    Q5perDegree(x, y, 10)

    x, y = generate_sine_data(80)
    Q5perDegree(x, y, 80)

    x, y = generate_sine_data(320)
    Q5perDegree(x, y, 320)

if __name__ == "__main__" :
    main()