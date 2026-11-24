# DAY 39: NN REGRESSION

import numpy as np

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print("Current Working Directory:", os.getcwd()) # binding the working directory to smallest folder the python file is in
# this will make it so it should work both in terminal and vsCode when i have neural-net-exp open

import warnings

# Suppress only runtime warnings
warnings.filterwarnings('ignore', category=RuntimeWarning) # warnings are annoying and they dont seem to be affecting stuff rn

X = np.random.uniform(0, 10, (1000, 2))

y = (
    7 * X[:, 0]
    + 4 * X[:, 1]
    + 10
    + np.random.normal(0, 2, 1000)
)

indices = np.arange(len(X))
np.random.shuffle(indices)

X = X[indices]
y = y[indices] # shuffling dataset

train_size = len(X) // 10 * 8

testing_X = X[train_size:]
testing_y = y[train_size:]

y = y[:train_size]
X = X[:train_size]

print(testing_X.shape, "testing x shape")

weights = np.zeros((X.shape[1],5)) # X.shape[1] is the amount of conditions there are

mean = X.mean(axis=0)
std = X.std(axis=0) # for normalizing

X_normalized = (X - mean) / std
print(X_normalized.shape)

def mse(y_true, y_pred):
    return np.mean((y_true-y_pred)**2)

def mae(y_true, y_pred):
    print(y_true.shape, y_pred.shape)
    return np.mean(np.abs(y_true-y_pred))

def softmax(x):
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)
# not really memorizing this formula stuff either

def relu(x):
    return np.maximum(0, x)

n=X.shape[0]
b = np.zeros(3)

def nn_regression(layer_sizes, iterations, X, y, learning_rate, lambda_, dropout_rate):
    n = X.shape[0]
    y = y.reshape(-1, 1) # reshaping y so that it has a shape of (n,1)
    # this is more of a numpy broadcasting thing

    weights = []
    biases = []
    weight_gradients = []
    bias_gradients = []

    for i in range(len(layer_sizes)-1):
        weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2/layer_sizes[i])) # using the pattern mentioned above here
        biases.append(np.zeros(layer_sizes[i+1])) # num_neurons corresponds to layer_sizes[i+1]
        # he initialization also, and biases now to 0, to hopefully stop nn from exploding with massive nn

        weight_gradients.append(0)
        bias_gradients.append(0)

    for i in range(iterations):
        activation = X

        activations_lst = [X] # initialized with X, because X is part of the activations
        logits_lst = []
        dropout_mask_lst = []

        # hidden layers
        for j in range(len(layer_sizes) - 2):
            logits = activation @ weights[j] + biases[j]
            logits_lst.append(logits)
            activation = relu(logits)

            mask = (np.random.rand(*activation.shape) > dropout_rate) / (1-dropout_rate)
            # creates boolean matrix of true and false, which == 1s and 0s so i can multiply with the activation
            # dividing by 1-dropout rate makes it so the average activation stays the same because the neurons that aren't
            # killed by the mask will grow larger for that iteration

            dropout_mask_lst.append(mask)
            activation = activation * mask
            # this removes the dropped out neurons and makes the non dropped out ones smaller

            activations_lst.append(activation)
            

        # output (will now only be one single value per val in X, instead of an array of vals)
        output = activation @ weights[-1] + biases[-1]

        concatenated_weights = np.array(weights[0].flatten())
        # this is all kinda chaos but its just so that i can find the mean of the weights
        # this was all only for the loss printing

        loss = mse(y, output)

        if i % 100 == 0:
            print("iteration:", i, "loss:", loss, "max weight", np.max(concatenated_weights))

        # backwards pass
        # backprop actually remained almost entirely unchanged, except for the error, because now it is just the gradient of the numerical
        # error rather than just the error of the probabilities

        # OUTPUT
        error = 2 * (output - y) / n

        weight_gradients = [None] * len(weights)
        bias_gradients = [None] * len(biases)

        for k in range(len(layer_sizes)-2):
            weight_gradient = (
                1/n
            ) * activations_lst[-k-1].T @ error

            weight_gradient += lambda_ * weights[-k-1]
            # L2 regularization, will push the weights to be closer towards 0.
            # if the weight is large, then the gradient for that weight will be large too
            # when subtraction happens, it will be closer to 0
            # vice versa if its large in the negative direction

            bias_gradient = (
                1/n
            ) * np.sum(error, axis=0)

            hidden_error = error @ weights[-k-1].T

            relu_gradient = (logits_lst[-k-1] > 0)
            dropout_gradient = dropout_mask_lst[-k-1]

            hidden_gradient = hidden_error * relu_gradient * dropout_gradient

            error = hidden_gradient

            weight_gradients[-k-1] = weight_gradient
            bias_gradients[-k-1] = bias_gradient

        bias_gradient = (
            1/n
        ) * np.sum(error, axis=0)

        weight_gradients[0] = (1/n) * X.T @ error
        weight_gradients[0] += lambda_ * weights[0] # forgot about this part, that meant l2 regularization wasn't getting to this part
        # (the first layer)
        bias_gradients[0] = bias_gradient

        for l in range(len(weights)):
            weights[l] -= learning_rate * weight_gradients[l]
            biases[l] -= learning_rate * bias_gradients[l]
    return (weights, biases)

def test_nn_regression(weights, biases, X, y):
    y = y.reshape(-1, 1) # reshaping the y because right now its a matrix of (200,) when the predictions are (200,1)
    # this messes up my MAE, and this is also the exact same thing i did in nn_multiple_hidden_layers anyway so that it wouldnt
    # mess up MSE

    # just if something weird happened
    if len(weights) != len(biases):
        raise ValueError(
            f"Length of weights and biases do not match, "
            f"{len(weights)} != {len(biases)}"
        )

    activation = X

    # hidden layer forward propogation
    for i in range(len(weights)-1):
        logits = activation @ weights[i] + biases[i]

        activation = relu(logits)

    # output
    output = activation @ weights[-1] + biases[-1]
    # now, the output is simply just the output logit because we don't need to do softmax and stuff

    test_mae = mae(y, output)
    return test_mae # maybe accuracy isnt the right term?

testing_X_normalized = (testing_X - mean) / std

weights, biases = nn_regression([2,8,8,1], 5000, X_normalized, y, 0.2, 0, 0)
# chose 0.1 for l2 because it produced values close to the original loss, so neither the old loss or the new 
# l2 regularization thing would have too disproportionate of an effect
# then i chose 0.01 because 0.1 was causing loss problems and the confusion matrix was kinda bad
# also, when i printed the max weight, i noticed that 0.1 lambda had a larger max weight than 0.01 max lambda
# which is kinda counterproductive so it was worse in both ways, max weight and cross entropy loss

test_mae = test_nn_regression(weights, biases, testing_X_normalized, testing_y)

print(f"MAE: {test_mae}")