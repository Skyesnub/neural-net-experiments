# DAY 21: STARTING NEURAL NETWORKS

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

learning_rate = 0.1
y_onehot = np.eye(5)[grade]
n=X.shape[0]
b = np.zeros(5)

NUM_NEURONS = 16

hidden_weights = np.random.randn(2,NUM_NEURONS) # the reason that random hidden weights are needed is to break symmetry
hidden_bias = np.random.randn(NUM_NEURONS) # they will give each neuron a fresh new starting point

output_weights = np.random.randn(NUM_NEURONS, 5)
output_bias = np.zeros(5) # the 4 exists because of matrix dimensions having to match with the matrix dimensions of hidden weights
# and bias, but the 5 is because there are 5 output classes (F,D,C,B,A)
# X = (8000,2) so hidden_logits = (8000,4) bc (8000,2)(2,4)

for i in range(10000):
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
    grade,
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
        print(i, loss)


testing_X_normalized = (testing_X - mean) / std

testing_hidden_logits = testing_X_normalized @ hidden_weights + hidden_bias

testing_hidden_activation = relu(testing_hidden_logits)

testing_output_logits = (
    testing_hidden_activation @ output_weights + output_bias
)

testing_probabilities = softmax(testing_output_logits)
testing_predictions = np.argmax(testing_probabilities, axis=1)

testing_grade[testing_scores >= 90] = 4   # A
testing_grade[(testing_scores >= 80) & (testing_scores < 90)] = 3   # B
testing_grade[(testing_scores >= 65) & (testing_scores < 80)] = 2   # C
testing_grade[(testing_scores >= 50) & (testing_scores < 65)] = 1   # D
testing_grade[testing_scores < 50] = 0    # F

accuracy = np.mean(testing_predictions == testing_grade)

print("Accuracy:", accuracy) # for troubleshooting

confusion_matrix = np.zeros((5,5),dtype=int)

for actual,predicted in zip(testing_grade, testing_predictions): # this loop will just add a 1 to wherever the predictions and actual
    confusion_matrix[actual,predicted] += 1 # grade was
# f should be on top a should be on bottom

print(confusion_matrix)



