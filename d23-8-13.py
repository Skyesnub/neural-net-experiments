# DAY 23: MORE NEURAL NET EXPERIMENTATION

import numpy as np

X = np.array([ # literally just XOR
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 1, 1, 0])

weights = np.zeros((X.shape[1],5)) # X.shape[1] is the amount of conditions there are (in this case just)
# 2

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

NUM_OUTPUTS = 2

learning_rate = 0.1
y_onehot = np.eye(NUM_OUTPUTS)[y]
n=X.shape[0]
b = np.zeros(NUM_OUTPUTS)

NUM_NEURONS = 5
# with 1 neuron, it always seems very certain of 1 of the 4 values with the probabilities, and it normally gets that value correct.
# in a rare case it can make every probability 0.5 and get all 0s
# with 2 neurons, it can often get it correctly and be very confident about the probabilities
# it can also often be completely wrong and unconfident with only like 66% confidence
# 3 is very similar to 2, but it can get into the same situation as 2
# 4 i would expect should do it, since now there are the same num neurons as outputs
# surprisingly not as good as I expected though, it gets it correctly and confidently like 90% of the time, but can get into situations
# where its a 50/50 probability or a 66/33 probability
# even when going up to 6 it can mess up in as low as 10 iterations in my example
# even with 8, it can mess up (it does sometimes take a few hundred attempts at a time though, like 606 iterations)
# finally doubled it to 16, will see how long it takes to mess up

iterations = 0
fails = 0

for i in range(1):
    iterations += 1

    hidden_weights = np.random.randn(2,NUM_NEURONS) # the reason that random hidden weights are needed is to break symmetry
    hidden_bias = np.random.randn(NUM_NEURONS) # they will give each neuron a fresh new starting point

    output_weights = np.random.randn(NUM_NEURONS, NUM_OUTPUTS)
    output_bias = np.zeros(NUM_OUTPUTS) # the 4 exists because of matrix dimensions having to match with the matrix dimensions of hidden weights
    # and bias, but the 5 is because there are 5 output classes (F,D,C,B,A)
    # X = (8000,2) so hidden_logits = (8000,4) bc (8000,2)(2,4)

    for i in range(2500): # changed to 2500 because efficiency and i don't think it needs all 10000
        # forward pass
        hidden_logits = X_normalized @ hidden_weights + hidden_bias # this is literally the same as before, except now using hidden weights
        # instead of regular weights
        # X is (8000, 2) and hidden_weights is (2,4), so hidden logits is (8000,4), meaning 8000 students, and the 4 neurons

        hidden_activation = relu(hidden_logits) # relu is there so that the neural network can do nonlinear things instead
        # of stacking linear things on top of other linear things
        # doesn't change dimensions

        output_logits = hidden_activation @ output_weights + output_bias # using the non-linearly represented input logits and
        # the output weights, and bias to make the output logits
        # hidden_activation is (8000,4) output_weights is (4,5), so output logits is (8000,5) meaning 8000 students 5 outputs

        probabilities = softmax(output_logits) # this is the same as in d20, except now there's the relu in the mix

        #loss
        loss = categorical_cross_entropy(
        y,
        probabilities,
        is_sparse=True
        )
        #print("Loss:", loss) # this is all the same as earlier

        #print("Actual:", grade[0])
        #print("Probabilities:", probabilities[0])
        #print("Prediction:", np.argmax(probabilities[0]))

        #backwards pass

        output_error = probabilities - y_onehot
        # output error will put negative numbers where the correct place was showing hey, you should put more weight here
        # and will put pos numbers where the wrong places are saying you should put less weight here

        output_weight_gradient = (
            1/n
        ) * hidden_activation.T @ output_error
        # hidden_activation.T = (4,8000) and output_error = (8000,5) so the output weight gradient is (4,5)
        # same shape as output weights
        # each row is one hidden neuron, each column is one output
        # the val [row, column] represents how much the weight should change by to reduce the loss

        output_bias_gradient = (
            1/n
        ) * np.sum(output_error, axis=0)
        # shape is (5,) matching the output bias shape

        hidden_error = output_error @ output_weights.T
        # output error is (8000,5), output weights.t is (5,4)
        # so hidden error is (8000,4) which is 8000 students and 4 neurons again!
        # this represents how strongly each neuron is connected to each student (basically in some cases one neuron changing)
        # may affect it much more than another neuron

        relu_gradient = (hidden_logits > 0) # this makes an 8000,4 matrix of 1s and 0s (T and F) depending on if the neuron was killed by
        # relu or not

        hidden_gradient = hidden_error * relu_gradient
        #print(hidden_gradient)
        # this blocks the error gradient if relu made it 0, meaning the error will be ignored in this case

        hidden_weight_gradient = (
            1/n
        ) * X_normalized.T @ hidden_gradient
        # now the hidden gradient and the original values influence the weight gradient
        #print(hidden_weight_gradient.shape)

        hidden_bias_gradient = (1/n) * np.sum(hidden_gradient, axis=0)

        output_weights -= learning_rate * output_weight_gradient
        output_bias -= learning_rate * output_bias_gradient

        hidden_weights -= learning_rate * hidden_weight_gradient
        hidden_bias -= learning_rate * hidden_bias_gradient

        if i % 100 == 0:
            pass # commented out rn for efficiency in the big loop
            #print(i, loss)

    hidden_logits = X_normalized @ hidden_weights + hidden_bias
    hidden_activation = relu(hidden_logits)
    testing_output_logits = (hidden_activation @ output_weights + output_bias) # the entire process of actually going through and getting
    testing_probabilities = softmax(testing_output_logits) # the predictions
    print("Probabilities:", testing_probabilities)
    testing_predictions = np.argmax(testing_probabilities, axis=1)

    print(testing_predictions) # gets it right every time! (i kinda would've expected that after 2500 iterations and 16 neurons...)

    if loss > 0.01:
        fails += 1
        print(f"something bad happened in {iterations} iterations.")

print(f"it failed (had high loss) {fails}/100 times.")

import matplotlib.pyplot as plt

# Create a grid covering the input space
x1 = np.linspace(-2, 2, 300)
x2 = np.linspace(-2, 2, 300)

grid_x1, grid_x2 = np.meshgrid(x1, x2)

# Turn the grid into a giant list of [x1, x2] points
grid = np.column_stack((
    grid_x1.ravel(),
    grid_x2.ravel()
))

# Run the grid through the SAME network
grid_hidden_logits = grid @ hidden_weights + hidden_bias
grid_hidden_activation = relu(grid_hidden_logits)

grid_output_logits = (
    grid_hidden_activation @ output_weights + output_bias
)

grid_probabilities = softmax(grid_output_logits)

# Get the predicted class for every point
grid_predictions = np.argmax(grid_probabilities, axis=1)

# Turn predictions back into the shape of the grid
grid_predictions = grid_predictions.reshape(grid_x1.shape)

# Plot the decision regions
plt.contourf(
    grid_x1,
    grid_x2,
    grid_predictions,
    levels=[-0.5, 0.5, 1.5],
    alpha=0.3
)

# Plot the actual XOR points
plt.scatter(
    X_normalized[:, 0],
    X_normalized[:, 1],
    c=y,
    edgecolors="black",
    s=100
)

plt.xlabel("X1")
plt.ylabel("X2")
plt.title("XOR Neural Network Decision Boundary")
plt.show()


