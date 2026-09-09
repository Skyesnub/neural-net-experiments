# DAY 33: PURPOSEFULLY CREATING OVERFITTING / DROPOUT + L2 HYPERPARAMETER SEARCHING

import numpy as np

NUM_STUDENTS = 1000

hours_studied = np.random.randint(2,11,size=NUM_STUDENTS) # reason im not doing 0-11 is because almost everyone would fail
hours_slept = np.random.randint(2,11,size=NUM_STUDENTS) # if that happened

indices = np.arange(NUM_STUDENTS)
np.random.shuffle(indices)

hours_studied = hours_studied[indices] # the reason i had to do all this was because i need them to be in the same order that they were before
hours_slept = hours_slept[indices] # it actually might have been fine if i shuffled them both randomly but im more comfortable with this

X = np.column_stack((hours_studied[:NUM_STUDENTS//20*14], hours_slept[:NUM_STUDENTS//20*14])) # this will be training X
validation_X = np.column_stack((hours_studied[NUM_STUDENTS//20*14:NUM_STUDENTS//20*17], hours_slept[NUM_STUDENTS//20*14:NUM_STUDENTS//20*17]))
testing_X = np.column_stack((hours_studied[NUM_STUDENTS//20*17:], hours_slept[NUM_STUDENTS//20*17:]))
TRUE_WEIGHTS = np.array([7,4])
TRUE_B = 10

distance = np.abs(X.T[0] + X.T[1] - 12)
true_score = 100 - distance**2 * 5 + np.random.randint(-5, 6, size=len(X))

#true_score = X @ TRUE_WEIGHTS + TRUE_B + np.random.randint(-15,16,size=NUM_STUDENTS//10*8)
# will multiply the hours studied and slept by the weights, add b, then add a bit of randomness

grade = np.zeros(NUM_STUDENTS//10*7,dtype=int) # starting with 0s, to plan to change the values in the next few lines of code
# this time separated

#testing_scores = testing_X @ TRUE_WEIGHTS + TRUE_B + np.random.randint(-15,16,size=NUM_STUDENTS//10*2) # these are the 200
# scores that the model will be testing

testing_distance = np.abs(
    testing_X.T[0] + testing_X.T[1] - 12
)
validation_distance = np.abs(
    validation_X.T[0] + validation_X.T[1] - 12
)

validation_scores = 100 - validation_distance**2 * 5 + np.random.randint(-5, 6, size=len(validation_X))
testing_scores = 100 - testing_distance**2 * 5 + np.random.randint(-5, 6, size=len(testing_X))

testing_grade = np.zeros(NUM_STUDENTS//20*3,dtype=int)
validation_grade = np.zeros(NUM_STUDENTS//20*3, dtype=int)

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

def neural_network_multiple_hidden_layers(layer_sizes, iterations, X, y, learning_rate, lambda_, dropout_rate):
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
        weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2/layer_sizes[i])) # using the pattern mentioned above here
        biases.append(np.zeros(layer_sizes[i+1])) # num_neurons corresponds to layer_sizes[i+1]
        # he initialization also, and biases now to 0, to hopefully stop nn from exploding with massive nn

        weight_gradients.append(0)
        bias_gradients.append(0)

    for i in range(iterations):
        activation = X

        #mask = (np.random.rand(*activation.shape) > dropout_rate) / (1-dropout_rate)
        #activation = activation * mask

        # this was input dropout, which isn't that wrong, but it removes parts of the data which is less important / a different
        # concept

        activations_lst = []
        logits_lst = []
        dropout_mask_lst = []

        #dropout_mask_lst.append(mask)

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

            activations_lst.append(activation)

        # output layer
        output_logits = activation @ weights[-1] + biases[-1]

        probabilities = softmax(output_logits)

        loss = categorical_cross_entropy(
            y,
            probabilities,
            is_sparse=True
        )

        concatenated_weights = np.array(weights[0].flatten())
        # this is all kinda chaos but its just so that i can find the mean of the weights
        for w in range(len(weights)-1):
            concatenated_weights = np.concatenate((concatenated_weights, weights[w+1].flatten()))

        weights_squared_mean = np.mean(concatenated_weights ** 2)
        l2_loss = weights_squared_mean * lambda_

        loss += l2_loss
        # this was all only for the loss printing

        if i % 100 == 0:
            print("iteration:", i, "loss:", loss, "max weight", np.max(concatenated_weights))

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

    print(len(weights), len(biases))
    return (weights, biases)

def test_my_neural_network(weights, biases, X, y, outputs):

    print("X finite:", np.all(np.isfinite(X)))
    print("X max:", np.max(np.abs(X)))

    if len(weights) != len(biases):
        raise ValueError(
            f"Length of weights and biases do not match, "
            f"{len(weights)} != {len(biases)}"
        )

    activation = X

    # hidden
    for i in range(len(weights)-1):
        logits = activation @ weights[i] + biases[i]

        """print("layer", i)
        print("logits max:", np.max(np.abs(logits)))
        print("logits finite:", np.all(np.isfinite(logits)))"""

        activation = relu(logits)

        """print("activation max:", np.max(np.abs(activation)))
        print("activation finite:", np.all(np.isfinite(activation)))"""

    # output
    output_logits = activation @ weights[-1] + biases[-1]

    """print("output logits max:", np.max(np.abs(output_logits)))
    print("output logits finite:", np.all(np.isfinite(output_logits)))"""

    probabilities = softmax(output_logits)

    """print("probabilities finite:", np.all(np.isfinite(probabilities)))  """

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

validation_grade[validation_scores >= 90] = 4
validation_grade[(validation_scores >= 80) & (validation_scores < 90)] = 3
validation_grade[(validation_scores >= 65) & (validation_scores < 80)] = 2
validation_grade[(validation_scores >= 50) & (validation_scores < 65)] = 1
validation_grade[validation_scores < 50] = 0


testing_X_normalized = (testing_X - mean) / std
validation_X_normalized = (validation_X - mean) / std


"""confusion_matrix = np.zeros((5,5),dtype=int)

for actual,predicted in zip(testing_grade, testing_predictions): # this loop will just add a 1 to wherever the predictions and actual
    confusion_matrix[actual,predicted] += 1 # grade was
# f should be on top a should be on bottom

print(confusion_matrix)"""

print("Actual:", np.bincount(testing_grade, minlength=5))

# multiple layers
weights, biases = neural_network_multiple_hidden_layers([2,32,16,8,5], 5000, X_normalized, grade, 0.2, 0, 0.1)
# chose 0.1 for l2 because it produced values close to the original loss, so neither the old loss or the new 
# l2 regularization thing would have too disproportionate of an effect
# then i chose 0.01 because 0.1 was causing loss problems and the confusion matrix was kinda bad
# also, when i printed the max weight, i noticed that 0.1 lambda had a larger max weight than 0.01 max lambda
# which is kinda counterproductive so it was worse in both ways, max weight and cross entropy loss

validation_accuracy, validation_confusion = test_my_neural_network(weights, biases, validation_X_normalized, validation_grade, 5)

print("Training accuracy:", 
      test_my_neural_network(weights, biases, X_normalized, grade, 5)[0])

print("Validation accuracy:",
      validation_accuracy)

print(validation_confusion)

#mult_layer_accuracy, mult_confusion = test_my_neural_network(weights, biases, testing_X_normalized, testing_grade, 5)

#print(f"Multiple layer accuracy: {mult_layer_accuracy}\nMulti layer confusion matrix: \n{mult_confusion}")


"""

Overall, at least for this problem, best dropout ~ 10% and best l2 ~ lambda_ = 0.001

It does overfit somewhat with the massive nn's, so I'm going to move on to dropout

DROPOUT NOTES:
dropout_rate = 0:
    Trial 1:
        Training: 0.95
        Validation: 0.91
    Trial 2:
        Training: 0.93
        Validation: 0.91
    Trial 3:
        Training: 0.94
        Validation: 0.9
dropout_rate = 0.1:
    Trial 1:
        Training: 0.92
        Validation: 0.95
    Trial 2:
        Training: 0.92
        Validation: 0.92
    Trial 3:
        Training: 0.92
        Validation: 0.92
dropout_rate = 0.2:
    Trial 1:
        Training: 0.91
        Validation: 0.94
    Trial 2:
        Training: 0.91
        Validation: 0.88
    Trial 3:
        Training: 0.93
        Validation: 0.91
dropout_rate = 0.5:
    Trial 1:
        Training: 0.92
        Validation: 0.89
    Trial 2:
        Training: 0.65
        Validation: 0.63
    Trial 3:
        Training: 0.92
        Validation: 0.93

L2 REGULARIZATION NOTES:

lambda_ = 0:
    Trial 1:
        Training: 0.93
        Validation: 0.92
    Trial 2:
        Training: 0.93
        Validation: 0.88
    Trial 3:
        Training: 0.93
        Validation: 0.91

lambda_ = 0.0001:
    Trial 1:
        Training: 0.93
        Validation: 0.95
    Trial 2:
        Training: 0.94
        Validation: 0.93
    Trial 3:
        Training: 0.94
        Validation: 0.89

lambda_ = 0.001
    Trial 1:
        Training: 0.92
        Validation: 0.91
    Trial 2:
        Training: 0.93
        Validation: 0.97
    Trial 3:
        Training: 0.94
        Validation: 0.92

lambda_ = 0.01 (my preferred one before this experiment)
    Trial 1:
        Training: 0.94
        Validation: 0.93
    Trial 2:
        Training: 0.93
        Validation: 0.93
    Trial 3:
        Training: 0.92
        Validation: 0.83

lambda_ = 0.1
    Trial 1:
        Training: 0.68
        Validation: 0.71
    Trial 2:
        Training: 0.67
        Validation: 0.71
    Trial 3:
        Training: 0.67
        Validation: 0.73

Combining:

dropout_rate = 0, lambda_ = 0
    Trial 1:
        Training: 0.91
        Validation: 0.92
    Trial 2:
        Training: 0.93
        Validation: 0.91
    Trial 3:
        Training: 0.93
        Validation: 0.9
dropout_rate = 0.1, lambda_ = 0
    Trial 1:
        Training: 0.93
        Validation: 0.91
    Trial 2:
        Training: 0.92
        Validation: 0.95
    Trial 3:
        Training: 0.93
        Validation: 0.9
dropout_rate = 0, lambda_ = 0.001
    Trial 1:
        Training: 0.91
        Validation: 0.9
    Trial 2:
        Training: 0.93
        Validation: 0.89
    Trial 3:
        Training: 0.93
        Validation: 0.91
dropout_rate = 0.1, lambda_ = 0.001
    Trial 1:
        Training: 0.91
        Validation: 0.89
    Trial 2:
        Training: 0.91
        Validation: 0.94
    Trial 3:
        Training: 0.92
        Validation: 0.91

"""