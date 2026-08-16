# DAY 25: GENERALIZING MY NEURAL NETWORK

import numpy as np

NUM_STUDENTS = 10000

hours_studied = np.random.randint(2,11,size=NUM_STUDENTS) # reason im not doing 0-11 is because almost everyone would fail
hours_slept = np.random.randint(2,11,size=NUM_STUDENTS) # if that happened

indices = np.arange(NUM_STUDENTS)
np.random.shuffle(indices)

hours_studied = hours_studied[indices] # the reason i had to do all this was because i need them to be in the same order that they were before
hours_slept = hours_slept[indices] # it actually might have been fine if i shuffled them both randomly but im more comfortable with this

X = np.column_stack((hours_studied[:NUM_STUDENTS//10*8], hours_slept[:NUM_STUDENTS//10*8])) # this will be training X
testing_X = np.column_stack((hours_studied[NUM_STUDENTS//10*8:], hours_slept[NUM_STUDENTS//10*8:]))
TRUE_WEIGHTS = np.array([7,4])
TRUE_B = 10

distance = np.abs(X.T[0] + X.T[1] - 12)
true_score = 100 - distance**2 * 5 + np.random.randint(-5, 6, size=len(X))

#true_score = X @ TRUE_WEIGHTS + TRUE_B + np.random.randint(-15,16,size=NUM_STUDENTS//10*8)
# will multiply the hours studied and slept by the weights, add b, then add a bit of randomness

grade = np.zeros(NUM_STUDENTS//10*8,dtype=int) # starting with 0s, to plan to change the values in the next few lines of code
# this time separated

#testing_scores = testing_X @ TRUE_WEIGHTS + TRUE_B + np.random.randint(-15,16,size=NUM_STUDENTS//10*2) # these are the 200
# scores that the model will be testing

testing_distance = np.abs(
    testing_X.T[0] + testing_X.T[1] - 12
)

testing_scores = 100 - testing_distance**2 * 5 + np.random.randint(-5, 6, size=len(testing_X))

testing_grade = np.zeros(NUM_STUDENTS//10*2,dtype=int)

grade[true_score >= 90] = 4   # A
grade[(true_score >= 80) & (true_score < 90)] = 3   # B
grade[(true_score >= 65) & (true_score < 80)] = 2   # C
grade[(true_score >= 50) & (true_score < 65)] = 1   # D
grade[true_score < 50] = 0    # F

print(grade.shape)

print(np.bincount(grade))

weights = np.zeros((X.shape[1],5)) # X.shape[1] is the amount of conditions there are (in this case just)
# hours studied and slept

mean = X.mean(axis=0)
std = X.std(axis=0) # for normalizing

X_normalized = (X - mean) / std
print(X_normalized.shape)

def softmax(x):
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)
# not really memorizing this formula stuff either

def categorical_cross_entropy(y_true, y_pred, is_sparse=False):
    """
    Computes the Categorical Cross-Entropy Loss using NumPy.
    
    Parameters:
    y_true (np.ndarray): Ground truth labels. 
                         If is_sparse=False, shape is (batch_size, num_classes).
                         If is_sparse=True, shape is (batch_size,).
    y_pred (np.ndarray): Predicted probabilities, shape (batch_size, num_classes).
    is_sparse (bool): Set to True if y_true contains integer class indices.
    
    Returns:
    float: Mean categorical cross-entropy loss over the batch.
    """
    # 1. Clip predictions to prevent log(0) which results in NaN or Infinity
    # We clip slightly above 0 and below 1
    y_pred = np.clip(y_pred, 1e-15, 1.0 - 1e-15)
    
    # 2. Compute loss based on target format
    if is_sparse:
        # Case 1: Sparse/Integer labels (e.g., [0, 2, 1])
        batch_size = len(y_true)
        # Use advanced indexing to pull the probabilities of the correct classes
        correct_confidences = y_pred[np.arange(batch_size), y_true]
        loss = -np.log(correct_confidences)
    else:
        # Case 2: One-Hot Encoded labels (e.g., [[1, 0, 0], [0, 0, 1]])
        # Formula: -sum(y_true * log(y_pred)) along the class axis
        loss = -np.sum(y_true * np.log(y_pred), axis=-1)
        
    # 3. Return the average loss across the entire batch
    return np.mean(loss)
# also not really memorizing this, but got a function that has a ton of comments!

def relu(x):
    return np.maximum(0, x)

y_onehot = np.eye(5)[grade]
n=X.shape[0]
b = np.zeros(5)

def neural_network_multiple_hidden_layers(layer_sizes, iterations, X, y, learning_rate):
    # layer sizes: [2,8,6,5] (example)
    # 2 is the inputs, 8 and 6 neurons, 5 outputs
    # weights[0] will be shape 2,8, weights[1] will be 8,6. pattern is [i], [i+1]

    n = X.shape[0]

    y_onehot = np.eye(layer_sizes[-1])[y] # layer_sizes[-1] gets the last layer size, which is the output

    weights = []
    biases = []
    weight_gradients = []
    bias_gradients = []

    for i in range(len(layer_sizes)-1):
        weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1])) # using the pattern mentioned above here
        biases.append(np.random.randn(layer_sizes[i+1])) # num_neurons corresponds to layer_sizes[i+1]

        weight_gradients.append(0)
        bias_gradients.append(0)

    for i in range(iterations):
        activation = X

        activations_lst = []
        logits_lst = []

        # hidden layers
        for j in range(len(layer_sizes) - 2):
            logits = activation @ weights[j] + biases[j]
            logits_lst.append(logits)
            activation = relu(logits)
            activations_lst.append(activation)

        # output layer
        output_logits = activation @ weights[-1] + biases[-1]

        probabilities = softmax(output_logits)

        loss = categorical_cross_entropy(
            y,
            probabilities,
            is_sparse=True
        )

        if i % 100 == 0:
            print(i, loss)

        # backwards pass

        # NEED FOR THIS PART:
        # need to save activations to a list
        # need to save logits to a list
        # then i can move on to generalizing backprop

        # OUTPUT
        output_error = probabilities - y_onehot
        error = output_error

        weight_gradients = [None] * len(weights)
        bias_gradients = [None] * len(biases)

        for k in range(len(layer_sizes)-2):
            weight_gradient = (
                1/n
            ) * activations_lst[-k-1].T @ error

            bias_gradient = (
                1/n
            ) * np.sum(error, axis=0)

            hidden_error = error @ weights[-k-1].T

            relu_gradient = (logits_lst[-k-1] > 0)

            hidden_gradient = hidden_error * relu_gradient

            error = hidden_gradient

            weight_gradients[-k-1] = weight_gradient
            bias_gradients[-k-1] = bias_gradient



        bias_gradient = (
            1/n
        ) * np.sum(error, axis=0)

        weight_gradients[0] = (1/n) * X.T @ error
        bias_gradients[0] = bias_gradient

        for l in range(len(weights)):
            weights[l] -= learning_rate * weight_gradients[l]
            biases[l] -= learning_rate * bias_gradients[l]

    print(len(weights), len(biases))
    return (weights, biases)


    
    
def neural_network_one_hidden_layer(num_neurons, iterations, X, y, num_outputs, learning_rate): # gonna assume X is normalized for this
    n = X.shape[0]
    num_inputs = X.shape[1]
    y_onehot = np.eye(num_outputs)[y]

    hidden_weights = np.random.randn(num_inputs, num_neurons)
    hidden_bias = np.random.randn(num_neurons)

    output_weights = np.random.randn(num_neurons, num_outputs)
    output_bias = np.zeros(num_outputs) # initialization

    for i in range(iterations):
        #forward pass
        hidden_logits = X @ hidden_weights + hidden_bias

        hidden_activation = relu(hidden_logits)

        output_logits = hidden_activation @ output_weights + output_bias

        probabilities = softmax(output_logits)

        loss = categorical_cross_entropy(
            y, probabilities, is_sparse=True
        )
        # backwards pass

        output_error = probabilities - y_onehot

        output_weight_gradient = (
            1/n
        ) * hidden_activation.T @ output_error

        output_bias_gradient = (
            1/n
        ) * np.sum(output_error, axis=0)

        hidden_error = output_error @ output_weights.T

        relu_gradient = (hidden_logits > 0)

        hidden_gradient = hidden_error * relu_gradient

        hidden_weight_gradient = (
                1/n
            ) * X.T @ hidden_gradient

        hidden_bias_gradient = (1/n) * np.sum(hidden_gradient, axis=0)

        output_weights -= learning_rate * output_weight_gradient
        output_bias -= learning_rate * output_bias_gradient

        hidden_weights -= learning_rate * hidden_weight_gradient
        hidden_bias -= learning_rate * hidden_bias_gradient

        if i % 100 == 0:
            print(i,loss)

    return (hidden_weights, hidden_bias, output_weights, output_bias)

def test_my_neural_network(weights, biases, X, y, outputs):

    if len(weights) != len(biases):
        raise ValueError(
            f"Length of weights and biases do not match, "
            f"{len(weights)} != {len(biases)}"
        )

    activation = X

    # hidden
    for i in range(len(weights) - 1):
        logits = activation @ weights[i] + biases[i]
        activation = relu(logits)

    # output
    output_logits = activation @ weights[-1] + biases[-1]

    probabilities = softmax(output_logits)

    predictions = np.argmax(probabilities, axis=1)

    accuracy = np.mean(predictions == y)

    confusion_matrix = np.zeros((outputs, outputs),dtype=int)

    for actual,predicted in zip(y, predictions):
        confusion_matrix[actual, predicted] += 1

    return accuracy, confusion_matrix


testing_grade[testing_scores >= 90] = 4
testing_grade[(testing_scores >= 80) & (testing_scores < 90)] = 3
testing_grade[(testing_scores >= 65) & (testing_scores < 80)] = 2
testing_grade[(testing_scores >= 50) & (testing_scores < 65)] = 1
testing_grade[testing_scores < 50] = 0 # kinda deleted this part and then realized i needed it


testing_X_normalized = (testing_X - mean) / std

"""confusion_matrix = np.zeros((5,5),dtype=int)

for actual,predicted in zip(testing_grade, testing_predictions): # this loop will just add a 1 to wherever the predictions and actual
    confusion_matrix[actual,predicted] += 1 # grade was
# f should be on top a should be on bottom

print(confusion_matrix)"""

print("Actual:", np.bincount(testing_grade, minlength=5))

# multiple layers
weights, biases = neural_network_multiple_hidden_layers([2,32,16,8,5], 10000, X_normalized, grade, 0.1)
mult_layer_accuracy, mult_confusion = test_my_neural_network(weights, biases, testing_X_normalized, testing_grade, 5)

hidden_weights, hidden_bias, output_weights, output_bias = neural_network_one_hidden_layer(
    16, 10000, X_normalized, grade, 5, 0.1
) # order: num_neurons, iterations, X, y, num_outputs, learning_rate
slweights = [hidden_weights, output_weights]
slbiases = [hidden_bias, output_bias]
single_layer_accuracy, sl_confusion = test_my_neural_network(slweights, slbiases, testing_X_normalized, testing_grade, 5)

print(f"Multiple layer accuracy: {mult_layer_accuracy}\nMulti layer confusion matrix: \n{mult_confusion}\nSingle layer accuracy: {single_layer_accuracy}\nSingle layer confusion:\n {sl_confusion}")



