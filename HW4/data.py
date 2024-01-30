""""
Use the linear model y = 2x + ε with zero-mean Gaussian noise ε ∼ N(0, 1) to
generate 500 data points with (equal spacing) x ∈ [-100, 100].
"""

import numpy as np
import matplotlib.pyplot as plt

def generate_data():
    epsilon = np.random.normal(0, 1, 500)
    x = np.linspace(-100, 100, 500)
    y = 2 * x + epsilon
    return x, y

def main():
    x, y = generate_data()

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.scatter(x, y)
    plt.title("Data Point")
    plt.show()

if __name__ == "__main__" :
    main()