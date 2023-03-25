import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib import cm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn import linear_model
from past.builtins import xrange
from scipy.special import expit

# TODO: Think this is only needed for Colab
# %matplotlib inline

def main():

    # Read in data
    data = pd.read_csv("../data/MSFT.csv")

    # Average between high and low values; what the stock is worth on average for any given day
    data['Average'] = (data['High'] + data['Low']) / 2

    # Add the column of ones to the data
    data.insert(0, 'Ones', 1)
    # Create a new column converting the date type into an integer, use these values for tracking since start of data
    data['Date'] = pd.to_datetime(data['Date'])
    data.insert(1, 'DateInt', range(0, len(data)))
    print(data)

    '''
    Separate the data into x and y:
        x = The ones column and ID (date as integer)
        y = The average of the High and Low values
    '''
    cols = data.shape[1]
    X = data[['Ones', 'DateInt']]
    Y = data['Average']
    X = np.asarray(X.values)
    Y = np.asarray(Y.values)
    theta = np.matrix(np.array([0, 0])).T
    print("X = ", X)
    print("X shape = ", X.shape)
    Y = Y.reshape(9083, 1)  # Not sure why it's the wrong shape
    print("Y = ", Y)
    # Reshape Y crom (9083, ) to (9083, 1)
    print("Y shape = ", Y.shape)
    print("Theta = ", theta)
    print("Theta shape = ", theta.shape)

    model = linear_model.LinearRegression(fit_intercept=False)
    model.fit(X, Y)
    model_coef = model.coef_
    print("Model Coefficients: ", model_coef)

    # Create prediction line
    f = model.predict(X).flatten()
    print("f = ", f)

    # Plot the data with the prediction line
    plt.scatter(data['Date'], data['Average'], label='Training Data')
    plt.plot(data['Date'], f, color="red", label='Predicted')
    plt.title("Microsoft Average Daily Stock Price (No adjustments)")
    plt.xlabel("Date")
    plt.ylabel("Average Daily Market Value")
    plt.legend()
    plt.show()

    def calcVectorizedCost(x, y, theta):
        inner = np.dot(((x * theta) - y).T, (x * theta) - y)
        return inner / (2 * len(x))
    print("Cost: ", calcVectorizedCost(X, Y, theta))

    # Create a gradient descent function that takes int the x, y, theta, eta, and iters as parameteres
    def gradientDescent(x, y, theta, eta, iters):
        cost = np.zeros(iters)
        # print("Theta Shape: ", theta.shape)
        # theta = theta.reshape(2, 1)
        # theta = np.expand_dims(theta, axis = -1)
        for i in range(iters):
            gradients = 2 * (np.dot(x.T, ((np.dot(x, theta))) - y) / (len(x)))
            theta = theta - eta * gradients

            cost[i] = calcVectorizedCost(x, y, theta)
        return theta, cost

    # Run the gradient descent function
    eta = 0.5
    iters = 40  # TODO: Anything above 40 produces optheta of [[nan][nan]]?
    optTheta, cost = gradientDescent(X, Y, theta, eta, iters)
    print("Optimal Theta: ", optTheta)
    # print("Cost: ", cost)
    print("Cost with Optimal Theta: ", calcVectorizedCost(X, Y, optTheta))


    '''
    Non-Linear Model 
    '''
    def generate_polynomial_features(X, degree):
        poly_features = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly_features.fit_transform(X)
        return X_poly

    def nonlinear_regression(X, y, degree):
        X_poly = generate_polynomial_features(X, degree)
        model = LinearRegression(fit_intercept=False)
        model.fit(X_poly, y)
        y_pred = model.predict(X_poly)
        return y_pred, model.coef_

    # Low degree -> Underfitted model (like 2)
    # High degree -> Overfitted model (like 10)
    degree = 4
    y_pred, coef = nonlinear_regression(X, Y, degree)

    # Plot the data with the predicted line
    plt.scatter(data['Date'], data['Average'], label='Training Data')
    plt.plot(data['Date'], y_pred, color="red", label='Predicted')
    plt.title(f"Microsoft Average Daily Stock Price (Degree {degree} polynomial)")
    plt.xlabel("Date")
    plt.ylabel("Average Daily Market Value")
    plt.legend()
    plt.show()

    print(f"Model Coefficients: {coef}")

    ''' 
    Non-Linear regression with Elastic Net
    '''
    def GeneratePolynomialFeatures(X, polydegree):
        poly = PolynomialFeatures(degree=polydegree)
        polynomial_x = poly.fit_transform(X)
        return polynomial_x


    # @param alpha: The weight of the regularization term
    # @param l1_ratio: The ratio of L1 regularization to L2 regularization
    def nonlinear_regression_elastic(X, y, degree, alpha, l1_ratio):
        X_poly = GeneratePolynomialFeatures(X, degree)
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False)
        model.fit(X_poly, y)
        y_pred = model.predict(X_poly)
        return y_pred, model.coef_

    # Fit the model using Elastic Net regularization
    # Mess with params, overall produces better fit after elastic net at same degree value
    y_pred, model_coef = nonlinear_regression_elastic(X, Y, degree=4, alpha=2, l1_ratio=0.5)

    # Plot the data and the predicted values
    plt.scatter(data['Date'], data['Average'], label='Training Data')
    plt.plot(data['Date'], y_pred, color='red', label='Predicted')
    plt.title(f"Microsoft Average Daily Stock Price (Elastic Net, Degree {degree})")
    plt.xlabel('Date')
    plt.ylabel('Average Daily Market Value')
    plt.legend()
    plt.show()


    '''
    Plotting for multiple degree values
    '''
    poly_degree_values = [2, 3, 4, 6, 8]
    plot_colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, degree in enumerate(poly_degree_values):
        y_pred, model_coef = nonlinear_regression_elastic(X, Y, degree=degree, alpha=2, l1_ratio=0.9)
        plt.plot(data['Date'], y_pred, color=plot_colors[i], label=f"Predicted (Degree {degree})")
        plt.title(f"Microsoft Average Daily Stock Price (Elastic Net)")
        plt.xlabel('Date')
        plt.ylabel('Average Daily Market Value')

    plt.scatter(data['Date'], data['Average'], color='#1f77b4', label='Training Data')  # If this is in the loop, it replaces a legend value
    plt.legend(poly_degree_values)
    plt.show()

    '''
    Classifying as "good" or "poor" performing stock
    '''
    # How many years of data to analyze
    num_years = 10
    recent_data = data.tail(252*num_years)

    # Compute the year-over-year percentage change in closing prices for each year
    yearly_pct_change = recent_data['Close'].pct_change(periods=252).groupby(recent_data['Date'].dt.year).mean()
    sum_pct_change = yearly_pct_change.sum()

    print(f"{num_years} Year Analysis: ")
    # Classify the stock as "good" or "poor" performing based on the trend
    if sum_pct_change > 0:
        print("The stock is performing well (good).")
    else:
        print("The stock is performing poorly (poor).")



if __name__ == "__main__":
    main()
