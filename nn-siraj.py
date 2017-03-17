# coding: utf-8

import numpy as np


# the sigmoid function, which is the type of non-linearity chosen for this neural net. 
# It is not the only type of non-linearity that can be chosen, but is has nice analytical features and is easy to teach with. 
# In practice, large-scale deep learning systems use piecewise-linear functions because they are much less expensive to evaluate. 
# 
# The implementation of this function does double duty. 
# If the deriv=True flag is passed in, the function instead calculates the derivative of the function, which is used in the error backpropogation step. 
def nonlin(x, deriv=False):
    if(deriv==True):
        return (x*(1-x))
    
    return 1/(1+np.exp(-x))


# input data
# The following code creates the input matrix. 
# The third column is for accommodating the bias term and is not part of the input. 
X = np.array([
            [0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])


# output data
# The output of the exclusive OR function follows. 
y = np.array([
            [0],
            [1],
            [1],
            [0]])

# The seed for the random generator which is useful for debugging
# deterministic
np.random.seed(1)


# synapses
# Now we intialize the weights to random values. 
# syn0 are the weights between the input layer and the hidden layer.
# It is a 3x4 matrix because there are two input weights plus a bias term (=3) 
# and four nodes in the hidden layer (=4). 
#
# syn1 are the weights between the hidden layer and the output layer. 
# It is a 4x1 matrix because there are 4 nodes in the hidden layer and one output. 
# Note that there is no bias term feeding the output layer in this example. 
# The weights are initially generated randomly because optimization tends not to work well when all 
# the weights start at the same value.
syn0 = 2*np.random.random((3,4)) - 1  # 3x4 matrix of weights ((2 inputs + 1 bias) x 4 nodes in the hidden layer)
syn1 = 2*np.random.random((4,1)) - 1  # 4x1 matrix of weights. (4 nodes x 1 output) - no bias term in the hidden layer.


# training step
# This is the main training loop. 
# The output shows the evolution of the error between the model and desired. 
# The error steadily decreases.  
for j in range(60000):  
    
    # Calculate forward through the network.
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))
    l2 = nonlin(np.dot(l1, syn1))
    
    # Back propagation of errors using the chain rule. 
    l2_error = y - l2
    if(j % 10000) == 0:   # Only print the error every 10000 steps, to save time and limit the amount of output. 
        print("Error: " + str(np.mean(np.abs(l2_error))))
        
    l2_delta = l2_error*nonlin(l2, deriv=True)
    
    l1_error = l2_delta.dot(syn1.T)
    
    l1_delta = l1_error * nonlin(l1,deriv=True)
    
    #update weights (no learning rate term)
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)
    
print("Output after training")
print(l2)
    
# See how the final output closely approximates the true output [0, 1, 1, 0]. 
# If you increase the number of interations in the training loop (currently 60000), 
# the final output will be even closer. 