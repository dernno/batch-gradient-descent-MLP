# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 19:24:48 2024

@author: dernno

One Layer Neural Network Model Implementation using batch gradient descent -
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

import random 

class Network():
    def initialize_parameters(self, sizes):   
        
        self.b = np.zeros(sizes[1]) #size (3,)
        self.W = 0.01 * np.random.randn(sizes[0], sizes[1])
    
    def ReLU(self, Z):
        # element wise
        return np.maximum(0, Z)
    
    def sigmoid(self, Z):
        # transforms z into range(0,1)
        return 1.0 / (1.0 + np.exp(-Z))
    
    def softmax(self, z):
        exp_values = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)
    
    
    def linear_forward(self, X):
        return np.dot(X, self.W) + self.b
    
    def activation_foward(self, Z, activation):
        
        if activation == "sigmoid":
            A = self.sigmoid(Z)
        elif activation == "relu":
            A = self.ReLU(Z)
        elif activation == "softmax":
            A = self.softmax(Z)
        
        return A
    
    def model_forward(self, X):
        Z = self.linear_forward(X)
        A = self.activation_foward(Z, "softmax")
        return A
    
    
    def compute_cost(self, output, y_target):
        lambda_val=0.01
        ce_loss = np.mean(self.cross_entropy(output, y_target))
        l2_regularization_cost = (lambda_val / (2 * len(y_target))) * np.sum(np.square(self.W))
        return ce_loss + l2_regularization_cost
    
    def cross_entropy(self, y_pred, y_target): 
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        return - np.log(np.sum(y_pred_clipped * y_target, axis=1))
    
    def model_backward(self, X, y, predictions): #mit model_state
        lambda_val=0.01
        gradient_weights = (-2 / len(y)) * np.dot(X.T,y -  predictions) + lambda_val * self.W
        gradient_bias = (-2 / len(y)) * np.sum(y- predictions, axis=0) #change dim
        return gradient_bias, gradient_weights
    
    def update_parameters(self, gradient_bias, gradient_weights, learning_rate):
        self.b -= learning_rate * gradient_bias
        self.W -= learning_rate * gradient_weights
        
    def predict(self, X):
        return self.model_forward(X)
    
    def shuffle_data(self, X, y):
        N, _ = X.shape
        shuffled_idx = np.random.permutation(N)
        return X[shuffled_idx], y[shuffled_idx]
    
    def get_predicted_class(self, y):
         return y.argmax(axis=1)
     
    def get_accuracy(self, predictions, y):
        #print(predictions, y)
        return np.mean(predictions == y)
    
    
    
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
                X_mini = X_train[batch_idx: batch_idx + batch_size] 
                y_mini = Y_train[batch_idx: batch_idx + batch_size] 
                
                predictions = self.predict(X_mini)
            
                train_cost = self.compute_cost(predictions, y_mini)
                train_accuracy = self.get_accuracy(self.get_predicted_class(predictions), self.get_predicted_class(y_mini))
                
                gradient_bias, gradient_weights = self.model_backward(X_mini, y_mini, predictions)
                
                self.update_parameters(gradient_bias, gradient_weights, learning_rate)
                
                    
            test_predictions = self.predict(X_test)
            test_cost = self.compute_cost(test_predictions, Y_test)
            test_accuracy = self.get_accuracy(self.get_predicted_class(test_predictions), self.get_predicted_class(Y_test))
            
            train_costs.append(train_cost)
            train_accuracies.append(train_accuracy)
            test_costs.append(test_cost)
            test_accuracies.append(test_accuracy)

            if iter % 20 == 0:
                print(f"Iteration {iter}: Train Loss = {train_cost}, Train Accuracy = {train_accuracy}, Test Loss = {test_cost}, Test Accuracy = {test_accuracy}")
        
        return np.array(train_costs), np.array(test_costs), np.array(train_accuracies), np.array(test_accuracies)
    
    def visualize_weights(self):
        fig, axs = plt.subplots(2, 5, figsize=(10, 4))  
        for i in range(10):
            # Reshape the i-th column of weights to 28x28
            weight_image = self.W[:, i].reshape(28, 28)
            
            ax = axs[i // 5, i % 5]  # Determine the position on the grid
            ax.imshow(weight_image, cmap='gray', interpolation='nearest')  # Use gray scale as these are intensity values
            ax.set_title(f'Neuron {i}')
            ax.axis('off')  # Turn off axis to make it cleaner
    
        plt.tight_layout()
        plt.savefig("visualize_weights.eps", format='eps')
        plt.show()
    
                        
if __name__ == '__main__':
    
    X_train, Y_train, X_test, Y_test = load_mnist()
    #X_train, Y_train, X_test, Y_test = load_mnist_reduced(500)
    
    # Parameter
    model_size = [784,10]
    num_iterations = 300
    learning_rate = 1e-2
    batch_size = 1000
    
    
    network = Network()
    train_costs, test_costs, train_accuracies, test_accuracies = network.train_model(X_train, Y_train, X_test, Y_test, model_size, num_iterations, learning_rate, batch_size)
    
    print()
    print("Optimized Weights:", network.W)
    print()
    print("Optimized Bias:", network.b)
    
    training_curve_plot("Training Curve", train_costs, test_costs, train_accuracies, test_accuracies)
    
    network.visualize_weights()
    
    
    
    
    

    