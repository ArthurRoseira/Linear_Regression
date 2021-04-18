import numpy as np
import random as rnd
import pandas as pd

class Network:
    def __init__(self, params, epochs, eta, x, y):
        self.layers = len(params)
        self.bias = 0
        self.topology = params
        self.weights = np.zeros(x.shape[1])
        self.output = [[0 for j in range(self.topology[i])]
                       for i in range(self.layers)]
        self.mse = []
        self.fit(x, y, epochs, eta)

    def fit(self, x, y, epochs, eta):
        for j in range(epochs):
            iteration_error = 0
            for sample in range(x.shape[0]):
                for layer in range(self.layers):
                    for neuron in range(self.topology[layer]):
                        self.output[layer][neuron] = self.feed_forward(
                            x.iloc[sample, :])
                iteration_output = self.output[-1]
                iteration_error = self.get_error(iteration_output, y.iloc[sample])    
                self.back_propagate(
                    x.iloc[sample, :], eta, iteration_output, y.iloc[sample])
            print('Error in Epoch ', j+1, ' : ', iteration_error[0])
            self.mse.append(self.MSE(x,y))


    def back_propagate(self, X, eta, y_calculated, y_actual):
        for i in range(len(y_calculated)):
            io = y_calculated[i]-0
            gradient = 2*(io - y_actual)
            for j in range(len(self.weights)):
                self.weights[j] = self.weights[j] - eta*gradient*X[j]
            self.bias = self.bias - eta*gradient

    def get_error(self, iteration_output, y_row):
        loss = []
        for i in range(len(iteration_output)):
            error = (iteration_output[i] - int(y_row))**2
            loss.append(error)
        return loss

    def MSE(self,X, y):
        error = 0
        for row in range(X.shape[0]):
            io = self.feed_forward(X.iloc[row,:])
            error = error + (io - y[row])**2
            mse = (1/(X.shape[0]))*error
        return mse

    def feed_forward(self, X_row):
        result = np.dot(self.weights, X_row) + self.bias
        return result

    def predict(self,X):
        y = np.dot(self.weights,X) + self.bias
        return y


if __name__ == '__main__':
    ds = pd.DataFrame({'x1': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'y': [
                      1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    y = ds.pop('y')
    X = ds
    a = Network([1], 1, 0.01, X, y)
