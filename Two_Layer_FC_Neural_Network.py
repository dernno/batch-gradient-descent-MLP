# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 19:24:48 2024

@author: dernno

Two Layer Neural Network Model Implementation using batch gradient descent -
used given load_mnist function to read and preprocess the MNIST dataset
    
The model consists of methods for initializing parameters, performing forward computation,
computing the cost (Cross Entropy), computing gradients, updating parameters, making predictions on mini-batches,
and training the model.
"""

import numpy as np
from load_mnist import load_mnist
from load_mnist_reduced import load_mnist_reduced
from training_curve_plot import training_curve_plot
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import random 

class Network():
    def initialize_parameters(self, sizes):   
        
        self.b1 = np.zeros(sizes[1]) 
        self.W1 = 0.01 * np.random.randn(sizes[0], sizes[1])
        
        self.b2 = np.zeros(sizes[2]) 
        self.W2 = 0.01 * np.random.randn(sizes[1], sizes[2])
        
        self.model_state = {}
    
    def ReLU(self, Z):
        # element wise
        return np.maximum(0, Z)
    
    def sigmoid(self, Z):
        # transforms z into range(0,1)
        return 1.0 / (1.0 + np.exp(-Z))
    
    def softmax(self, z):
        exp_values = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)
    
    
    def linear_forward(self, X, W, b):
        return np.dot(X, W) + b
    
    def activation_foward(self, Z, activation):
        
        if activation == "sigmoid":
            A = self.sigmoid(Z)
        elif activation == "relu":
            A = self.ReLU(Z)
        elif activation == "softmax":
            A = self.softmax(Z)
        
        return A
    
    def model_forward(self, X):
        
        Z1 = self.linear_forward(X, self.W1, self.b1)
        A1 = self.activation_foward(Z1, "relu")
        
        Z2 = self.linear_forward(A1 , self.W2, self.b2)
        A2 = self.activation_foward(Z2, "softmax")
        
        self.model_state['Z1'] = Z1
        self.model_state['A1'] = A1
        self.model_state['Z2'] = Z2
        self.model_state['A2'] = A2
        
        return A2
    
    def compute_cost(self, output, y_target):
        lambda_val=0.01
        ce_loss = np.mean(self.cross_entropy(output, y_target))
        l2_regularization_cost = (lambda_val / (2 * len(y_target))) * (np.sum(np.square(self.W1)) + np.sum(np.square(self.W2)))
        return ce_loss + l2_regularization_cost
    
    def linear_backward(self, y, predictions):
        return y -  predictions
    
    def sigmoid_backward(self, Z):
        return self.sigmoid(Z)*(1-self.sigmoid(Z))
    
    def relu_backward(self, Z):
        return Z > 0
    

    def model_backward(self, X, y, predictions):
        
        dZ2 = predictions - y
        dW2 = np.dot(self.model_state['A1'].T, dZ2) / len(y)
        db2 = np.sum(dZ2, axis=0) / len(y)
        
        dZ1 = np.dot(dZ2, self.W2.T) * self.relu_backward(self.model_state['Z1'])
        dW1 = np.dot(X.T, dZ1) / len(y)
        db1 = np.sum(dZ1, axis=0) / len(y)
        
        return db1, dW1, db2, dW2
    
    def update_parameters(self, db1, dW1, db2, dW2, learning_rate):
        
        self.b1 -= learning_rate * db1
        self.W1 -= learning_rate * dW1
        
        self.b2 -= learning_rate * db2
        self.W2 -= learning_rate * dW2
        
    def predict(self, X):
        
        return self.model_forward(X)
    
    def shuffle_data(self, X, y):
        N, _ = X.shape
        shuffled_idx = np.random.permutation(N)
        return X[shuffled_idx], y[shuffled_idx]
    
    def get_predicted_class(self, y):
         return y.argmax(axis=1)
     
    def get_accuracy(self, predictions, y):
        return np.mean(predictions == y)
    
    def cross_entropy(self, y_pred, y_target): 
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        return - np.log(np.sum(y_pred_clipped * y_target, axis=1))
    
    def train_model(self, X_train, Y_train, X_test, Y_test, model_size, num_iterations, learning_rate, batch_size):
            
        self.initialize_parameters(model_size)
        
        train_costs = []
        train_accuracies = []
        test_costs = []
        test_accuracies = []

        N, D = X_train.shape
        num_batches = N//batch_size
        
        for iter  in range(num_iterations):
            X_train, Y_train = self.shuffle_data(X_train,Y_train)
            
            for batch_idx in range(0, N, batch_size):
                #batch_number = (batch_idx // batch_size) + 1
                X_mini = X_train[batch_idx: batch_idx + batch_size] 
                y_mini = Y_train[batch_idx: batch_idx + batch_size] 
                
                predictions = self.predict(X_mini)
                
        
                train_cost = self.compute_cost(predictions, y_mini)
                train_accuracy = self.get_accuracy(self.get_predicted_class(predictions), self.get_predicted_class(y_mini))
                
                db1, dW1, db2, dW2 = self.model_backward(X_mini, y_mini, predictions)
                
                self.update_parameters(db1, dW1, db2, dW2, learning_rate)
                

                    
            test_predictions = self.predict(X_test)
            test_cost = self.compute_cost(test_predictions, Y_test)
            test_accuracy = self.get_accuracy(self.get_predicted_class(test_predictions), self.get_predicted_class(Y_test))
            
            train_costs.append(train_cost)
            train_accuracies.append(train_accuracy)
            test_costs.append(test_cost)
            test_accuracies.append(test_accuracy)

            if iter % 20 == 0:
                print(f"Iter {iter}: Train Loss = {train_cost:.3f}, Train Acc = {train_accuracy:.3f}, Test Loss = {test_cost:.3f}, Test Acc = {test_accuracy:.3f}")
        
        return np.array(train_costs), np.array(test_costs), np.array(train_accuracies), np.array(test_accuracies)
    
    
                        
if __name__ == '__main__':
    
    #X_train, Y_train, X_test, Y_test = load_mnist()
    X_train, Y_train, X_test, Y_test = load_mnist_reduced(60)
    #X_train, Y_train, _ , _ = load_mnist(5)
    
    # Parameter
    model_size = [784, 50, 10]
    num_iterations = 2000
    learning_rate = 1e-1
    batch_size = 1000
    
    
    network = Network()
    train_costs, test_costs, train_accuracies, test_accuracies = network.train_model(X_train, Y_train, X_test, Y_test, model_size, num_iterations, learning_rate, batch_size)
    
    print()
    print("Optimized Weights_1:", network.W1)
    print()
    print("Optimized Bias_1:", network.b1)
    
    print()
    print("Optimized Weights_2:", network.W2)
    print()
    print("Optimized Bias_2:", network.b2)
    
    training_curve_plot("Training Curve", train_costs, test_costs, train_accuracies, test_accuracies)
    
    
    
    
    
    

    