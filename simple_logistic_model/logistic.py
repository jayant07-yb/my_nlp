"""
Idea here is to test the performance of the different methods of iterating and finding the perfect learning rate.
Datas:
x --> integer
y --> integer

Let's try predicting the value of "y = (x-2000)^2 + 1000"
For range of x = 0 to 1000

Questions to be asked here:
q.1 What is the best way to iterate over the data?
q.2 Can dataset be divided into batches, with different weights for each batch?
q.3 What is the best way to find the weights?
q.4 What is the best way to find the best learning rate?
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import sys
import os
from math import sqrt

sys.path.append(os.path.abspath('bigram'))
from deep_learning import get_training_data


class LinearRegression():
    def __init__(self, learning_rate=0.1, output_file='large_files/output', variable_learning_rate=True):
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None
        self.output_file = output_file
        self.weight_path = os.path.join(self.output_file, 'linear_regression_weights.txt.npy')
        self.bias_path = os.path.join(self.output_file, 'linear_regression_bias.txt.npy')
        self.variable_learning_rate = variable_learning_rate

        # if os.path.exists(self.weight_path):
        self.weights = np.load(self.weight_path)
        # if os.path.exists(self.bias_path):
        self.bias = np.load(self.bias_path)

    def fit(self, X, Y, epochs=10, show_fig=False):
        # Initialize the weights and bias
        if self.weights is None:
            self.weights = np.random.randn(X.shape[1])
        
        if self.bias is None:
            self.bias = np.random.randn()

        """
        Lets store it before we start training
        """
        np.save(self.weight_path, self.weights)
        np.save(self.bias_path, self.bias)

        losses = []

        for epoch in range(epochs):
            
            np.save(self.weight_path, self.weights)
            np.save(self.bias_path, self.bias)
            # Calculate the prediction
            Yhat = self.predict(X)
            # Calculate the loss
            loss = self.loss(Y, Yhat)

            #What should be the learning rate?
            if self.variable_learning_rate and len(losses) > 0:
                """
                    Ideally loss should be decreasing, since we are aimig to towards down the slop.
                    Whats the problem with very large learning rate???
                    What if we get at the other side of the slop???
                        Since as we get close to the center of the parabola changes for this happening
                        will increase.

                        So if we crosses the center then we will have to decrease the learning rate.
                        If we are far away from the center then we can increase the learning rate.

                    If the loss is decreasing, then increase the learning rate
                    If the loss is increasing, then decrease the learning rate

                """
                if losses[-1] < loss:
                    self.learning_rate /= 2
                elif epoch % 4 == 0:
                    """
                    With time the learning rate should be decreased"""
                    self.learning_rate *= 1.25

            # Update the weights
            self.weights -= self.learning_rate * X.T.dot(Yhat - Y) / (X.shape[0])
            self.bias -= self.learning_rate * (Yhat - Y).sum() / (X.shape[0])

            losses.append(loss)

            print('Epoch: %d, Loss: %f' % (epoch, loss))

        if show_fig:
            plt.plot(losses)
            plt.show()

        np.save(self.weight_path, self.weights)
        np.save(self.bias_path, self.bias)

    def score(self, X, Y):
        Yhat = self.predict(X)
        
        return sqrt(((Y - Yhat)**2).mean())/sqrt((Y**2).mean())
    
    def predict(self, X):
        return X.dot(self.weights) + self.bias
    
    def loss(self, Y, Yhat):
        return ((Y - Yhat)**2).mean()
        

if __name__ == '__main__':
    liner_regression = LinearRegression()
    
    trainX, testX, trainY, testY = get_training_data()
    """
    Before actual trainign, lets try to find out whether th model is actually working or not???
    trainX = np.random.random((1000, 100))
    trainY = trainX.dot(np.random.random(100)) + np.ones(1000)*10
    """


    liner_regression.fit(trainX, trainY, epochs=10, show_fig=True)
    score = liner_regression.score(testX, testY)
    print(score)