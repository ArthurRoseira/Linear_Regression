import numpy as np
import random as rnd
import pandas as pd

# https://colab.research.google.com/drive/1f84s4nlKSas5LGpR8zdRxWOsKL5HIoyy
class Network:
    def __init__(self, params, epochs, eta, x, y):
        self.layers = len(params)
        self.bias = 0
        self.topology = params
        self.weights = np.zeros(x.shape[1])
        self.output = [[0 for j in range(self.topology[i])]
                       for i in range(self.layers)]
        self.fit(x, y, epochs, eta)

    def fit(self, x, y, epochs, eta):
        for j in range(epochs):
            iter_error = 0
            for sample in range(x.shape[0]):
                for layer in range(self.layers):
                    for neuron in range(self.topology[layer]):
                        self.output[layer][neuron] = self.feed_forward(
                            x.iloc[sample, :])
                iteration_output = self.output[-1]
                iteration_error = self.get_error(iteration_output, y.iloc[row])             
    def get_error(self, iteration_output, y_row):
        loss = []
        for i in range(len(iteration_output)):
            error = (iteration_output[i] - int(y_row))**2
            loss.append(error)
        return loss

    def get_mse(self,X, y):
        error = 0
        for row in range(X.shape[0]):
            io = self.feed_forward(X.iloc[row,:])
            error = error + (io - y[row])**2
            mse = (1/(X.shape[0]))*error
        return mse

    def feed_forward(self, X_row):
        result = self.bias + np.dot(self.weights, X_row)
        return result

    def predict(self,X):
        y=0
        for i in range(len(self.weights)):
            y= y + self.weights[i]*X[i]
        y=y+self.bias
        return y


if __name__ == '__main__':
    ds = pd.DataFrame({'x1': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'y': [
                      1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    y = ds.pop('y')
    X = ds
    a = Network([1], 1, 1, X, y)
    print(a.output)
