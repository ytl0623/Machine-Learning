""""
Change the model to y = sin(2πx) + ε with the noise ε ∼ N(0, 0.04) and (equal
spacing) x ∈ [0, 1]. Then repeat those stated in 2) and 3). Compare the results with
linear/polynomial regression on different datasets.
"""

import numpy as np
import math
import matplotlib.pyplot as plt

def generate_sine_data():
    epsilon = np.random.normal(0, 0.04, 15)
    x = np.linspace(0, 1, 15)
    y = np.sin(2 * x * np.pi) + epsilon
    return x, y

def linear_regression(x, y):
    x = np.column_stack((x, np.ones(x.shape[0])))  # 將x右邊填滿 1 column的 1
    y = y[:, np.newaxis]
    pin = np.linalg.pinv(x)  # 藉由np.linalg.pinv求得x的pseudo-inverse
    result = pin @ y  # 將pseudo-inverse與y相乘
    return result

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

def kfold(x, y, k):
    begin = 0
    terr = 0
    size = x.size

    for _ in range(k):
        x_train,x_test = kfold_split(x, begin, 1/k)
        y_train,y_test = kfold_split(y, begin, 1/k)
        wlin = linear_regression(x_train, y_train)
        pred = wlin[0] * x_test + wlin[1]
        err = training_error(pred, y_test)
        begin += math.floor(size * 1/k)
        terr += err

    return terr / k

def main():
    x, y = generate_sine_data()

    wlin = linear_regression(x, y)
    pred = wlin[0] * x + wlin[1]

    print("Q4 Linear Regression Training Error: " + str(training_error(pred, y)) + "\n")

    print("Q4 Linear Regression Five-Fold Cross Validation Error: " + str(kfold(x, y, 5)) + "\n")

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.scatter(x, y, label="Training Data")
    plt.title("Q4: Linear Regression")
    pred = pred.reshape(15, 1)
    plt.plot(x, pred, "r-", label="Fitting Function")
    plt.legend()
    plt.show()

if __name__ == "__main__" :
    main()