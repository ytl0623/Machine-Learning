""""
Following 4), perform polynomial regression of degree 14 via regularization.
Compare the results by setting λ = 0, 0.001/m , 1/m, 1000/m , where m = 15 is the
number of data points (with x = 0, 1/(m−1) , 2/(m−1), . . . , 1). Show the five-fold
cross-validation errors and the fitting plots.
"""

import numpy as np
import math
import matplotlib.pyplot as plt

def generate_sine_data():
    epsilon = np.random.normal(0, 0.04, 15)
    x = np.linspace(0, 1, 15)
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

def poly_compute(x , y , lam):
    wlin = np.matmul(np.matmul (np.linalg.inv(np.matmul(x.T , x) + lam * np.identity(x.shape[1])) , x.T) , y)
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

def kfold(x, y, k, degree, lam):
    begin = 0
    terr = 0
    size = x.size

    for _ in range(k):
        x_train,x_test = kfold_split(x, begin, 1/k)
        y_train,y_test = kfold_split(y, begin, 1/k)
        x_2 = polynomial_regression(x, degree)
        wlin = poly_compute(x_2, y, lam)

        pred = wlin[0] * x_test + wlin[1]
        err = training_error(pred, y_test)
        begin += math.floor(size * 1/k)
        terr += err

    return terr / k

def Q6perLamda(x, y, lam):
    degree = 14
    m = 15
    #x_train, x_test = splitdata(x)
    #y_train, y_test = splitdata(y)
    x_2 = polynomial_regression(x, degree)
    wlin = poly_compute(x_2, y, lam)
    print("Q6 λ = %f, Degree %d Polynomial Regression" % (lam, degree))
    p = np.poly1d(wlin.flatten())
    pred = p(x)

    print("Q6 Polynomial Regression Training Error: " + str(training_error(pred, y)))
    print("Q6 Polynomial Regression Five-Fold Cross Validation Error: " + str(kfold(x, y, 5, degree, lam)) + "\n")


    plt.xlabel("X")
    plt.ylabel("Y")
    plt.scatter(x, y, label="TrainData")
    plt.title("Q6: Polynomial Regression Degree %d, λ = %f " % (degree, lam))
    plt.plot(x, pred, "r-", label="Fitting Function")
    plt.legend()
    plt.show()

    return



def main():
    m = 15
    x, y = generate_sine_data()

    Q6perLamda(x, y, 0)  # lambda: 0
    Q6perLamda(x, y, 0.001 / m)  # lambda: 0.001/m
    Q6perLamda(x, y, 1 / m)  # lambda: 1/m
    Q6perLamda(x, y, 1000 / m)  # lambda: 1000/m

if __name__ == "__main__" :
    main()

