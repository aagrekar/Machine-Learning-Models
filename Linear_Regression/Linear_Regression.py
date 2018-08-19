
# coding: utf-8

# # Simple Linear Regression
#
# This code is an implementation of Simple Linear Regression using Gradient Descent method.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_data(X,y):
    """ This function generates a scatter plot.

    Parameters
    ----------
    X: np.array
        Vector containing the values for X-axis.
    y: np.array
        Vector containing the values for Y-axis.
    """
    plt.figure(figsize=(8,6))
    plt.scatter(X,y, marker="x", c="r", alpha=0.5)
    plt.show()

def computeCost(X, y, theta):
    """ This function computes the cost for Simple Linear Regression using theta as the parameters for Linear Regression
    to fit the data points in X and y.

    Parameters
    ----------
    X: np.array
        Matrix containing the X features with a column of ones (for intercept) appended in the beginning.
    y: np.array
        Vector containing the dependent variable column.
    theta: np.array
        Vector containing the theta parameters.

    Returns
    -------
    J: float
        Gradient value for the given X, y and theta.

    """
    m = len(y)   # set the number of samples.
    J = 0        # initialize the return variable.

    y_predicted = np.dot(X, theta)
    squared_error = (y - y_predicted)**2
    J = (sum(squared_error)/(2*m))[0]
    return J

def gradientDescent(X, y, theta, alpha, num_iter):
    """ This function performs Gradient Descent to learn the theta parameters by
        taking gradient steps with learning rate alpha.

    Parameters
    ----------
    X: np.array
        Matrix containing the X features with a column of ones (for intercept) appended in the beginning.
    y: np.array
        Vector containing the dependent variable column.
    theta: np.array
        Vector containing the theta parameters.
    alpha: float
        Learning Parameter to define the step size.
    num_iter: int
        Number of iterations to perform before the algorithm stops if no convergence reached.

    """
    m = len(y)       # set the number of samples
    J_history = []   # list to hold all the cost function values.
    for i in range(num_iter):
        y_predicted = np.dot(X, theta)
        theta = theta - alpha*(np.dot(X.T, y_predicted - y))/m
        J_history.append(computeCost(X, y, theta))
        if i > 0:
            if J_history[-1] > J_history[-2]:
                break

    return theta


# In[167]:


def main():
    # load data
    data = pd.read_table("../datasets/ex1data1.txt", header=None, sep=",")
    X = np.array([data[0].values]).T
    y = np.array([data[1].values]).T

    # Plot data
    plot_data(X, y)

    # Add the intercept Column of 1s
    X = np.append(np.ones(shape=(97,1)), X,1)
    theta = np.zeros(shape=(2,1))

    print("Testing the cost function ...")
    J = computeCost(X, y, theta)
    print("With theta as [[0],[0]] the cost computed = ", J)
    print("Expected Cost Value (approx) = 32.07 \n")

    # Further testing of the cost computing function
    J = computeCost(X, y, np.array([[-1],[2]]))
    print("With theta as [[-1],[2]] the cost computed = ", J)
    print("Expected Cost Value (approx) = 54.24")

    # Some gradient descent settings
    alpha = 0.01
    num_iter = 1500
    print("Runnint gradient descent ...")
    theta = gradientDescent(X, y, theta, alpha, num_iter)
    print("Theta found by gradient descent:")
    print(theta)
    print("Expected theta values (approx):")
    print(" -3.6303 \n 1.1664 \n")

    plt.figure(figsize=(8,6))
    plt.plot(X[:,1], y, 'x', X[:,1], np.dot(X,theta))
    plt.show()

if __name__ == '__main__':
    main()
