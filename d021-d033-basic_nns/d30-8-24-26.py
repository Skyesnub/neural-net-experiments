# DAY 30: ADAM

import numpy as np
import matplotlib.pyplot as plt

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

def neural_network_multiple_hidden_layers(layer_sizes, iterations, X, y, learning_rate, decay1, decay2, use_He=True):
    # layer sizes: [2,8,6,5] (example)
    # 2 is the inputs, 8 and 6 neurons, 5 outputs
    # weights[0] will be shape 2,8, weights[1] will be 8,6. pattern is [i], [i+1]

    iters = 0
    n = X.shape[0]

    y_onehot = np.eye(layer_sizes[-1])[y] # layer_sizes[-1] gets the last layer size, which is the output

    weights = []
    biases = []
    weight_gradients = []
    bias_gradients = []
    loss_lst = []

    weight_velocities = []
    bias_velocities = []
    squared_gradient_averages = []
    bias_squared_gradient_averages = []

    for i in range(len(layer_sizes)-1):
        if (use_He):
            weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2 / layer_sizes[i])) 
            biases.append(np.zeros(layer_sizes[i+1])) # He initialization is typically only applied to the weights
            # of a neural nework
            # it seems to help a lot with the random initialization of a neural network and keeping that more controlled
            # will prevent really massive nn's from dying due to random initialization
            # reason it uses 2 is because roughly half the neurons are calcelled out by ReLU
            # it kinda makes me think that its just outting the randomness and contricting it into a smaller area
            # (that's actually correct but its more than that), bc its precisely based on the layer size


            weight_velocities.append(
                np.zeros((layer_sizes[i], layer_sizes[i+1]))
            )

            bias_velocities.append(
                np.zeros(layer_sizes[i+1])
            )
        else:
            weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1])) # old one
            biases.append(np.random.randn(layer_sizes[i+1])) # no He

            weight_velocities.append(
                np.zeros((layer_sizes[i], layer_sizes[i+1]))
            )

            bias_velocities.append(
                np.zeros(layer_sizes[i+1])
            )

        weight_gradients.append(0)
        bias_gradients.append(0)

    for w in weights:
        squared_gradient_averages.append(np.zeros_like(w))
    for b in biases:
        bias_squared_gradient_averages.append(np.zeros_like(b))
    weight_moments = [np.zeros_like(w) for w in weights]
    bias_moments = [np.zeros_like(b) for b in biases]

    for i in range(iterations):
        iters += 1
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
            print("iterations:", i,
                  "loss:", loss,
                  "learning rate:", learning_rate,
                  )

        loss_lst.append(loss)

        """try:
            if loss_lst[i] + 0.001 > loss_lst[i-100]: # if the loss has improved by less than 0.002 over 100 iterations
                learning_rate *= 0.999 # then learning rate becomes smaller
        except IndexError:
            pass""" # commented out for no, might be breaking adam

        # backwards pass

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
            # ---MOMENTS---

            weight_moments[l] = (
                decay1 * weight_moments[l]
                + (1 - decay1) * weight_gradients[l]
            ) # now using weight and bias velocities
            # the velocities will be updated every iteration and will essentially measure which direction the 
            # gradient was going in the past for that weight
            # then it will add that to the weights
            # this still takes into account the gradients but ALSO the momentum

            bias_moments[l] = (
                decay1 * bias_moments[l]
                + (1 - decay1) * bias_gradients[l]
            )

            #this is now using bias correction, essentially as iters increases, it will start
            # to remember more and more from the previous iterations, because the iters value is increasing over time.
            # this means that decay1 ^ iters will constantly get smaller and smaller, meaning that the bias moment over time
            # will get larger and larger

            # ---CORRECTED MOMENTS---
            weight_moment_corrected = (
                weight_moments[l] / (1 - decay1**iters)
            )

            bias_moment_corrected = (
                bias_moments[l] / (1 - decay1**iters)
            )

            # ---SECOND MOMENTS---
            squared_gradient_averages[l] = (
                decay2 * squared_gradient_averages[l]
                + (1 - decay2) * weight_gradients[l]**2
            )

            bias_squared_gradient_averages[l] = (
                decay2 * bias_squared_gradient_averages[l]
                + (1-decay2) * bias_gradients[l]**2
            )

            # ---SECOND MOMENTS CORRECTED---
            weight_squared_corrected = (
                squared_gradient_averages[l]
                / (1 - decay2**iters)
            )

            bias_squared_corrected = (
                bias_squared_gradient_averages[l]
                / (1 - decay2**iters)
            )

            # ---UPDATE---
            weights[l] -= learning_rate * (
                weight_moment_corrected
                / (np.sqrt(weight_squared_corrected[l]) + 1e-3)
            ) # final update: first moment is essentially the direction, while the 2nd moment (squared_gradient_averages)
            # is essentially the magnitude. epsilon is to prevent division by 0

            biases[l] -= learning_rate * (
                bias_moment_corrected
                / (np.sqrt(bias_squared_corrected[l]) + 1e-3)
            )

            if not np.all(np.isfinite(weight_moments[l])):
                print("BAD WEIGHT MOMENT", l)

            if not np.all(np.isfinite(squared_gradient_averages[l])):
                print("BAD SQUARED GRADIENT", l)

            if not np.all(np.isfinite(weight_squared_corrected)):
                print("BAD CORRECTED SQUARED GRADIENT", l)

        if i % 1000 == 0:
            for m in range(len(activations_lst)):
                dead_amount = np.mean(activations_lst[m] == 0)
                print(f"Layer {m}: Dead %: {dead_amount*100}")
        # i noticed one case where between 87.5 and 100% of neurons were dead, and it caused the loss to be quite high
        # and it failed B's
        # this was with He, which i found strange because I thought He was supposed to fix the random initialiation problems
        # actually i used activations[m] instead of activations_lst[m] which gave me the percentage of neurons active for final layer
        # for each student



    print(len(weights), len(biases))
    return (weights, biases, loss_lst)


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

        print(
            "Layer", i,
            "max activation:", np.max(np.abs(logits)),
            "finite:", np.all(np.isfinite(logits)),
            "max weights:", np.max(weights[i]),
            "max biases:", np.max(biases[i])
        )

    # output
    output_logits = activation @ weights[-1] + biases[-1]

    probabilities = softmax(output_logits)

    predictions = np.argmax(probabilities, axis=1)

    accuracy = np.mean(predictions == y)

    confusion_matrix = np.zeros((outputs, outputs),dtype=int)

    print(
        "Final activation max:", np.max(np.abs(activation)),
        "Final weights max:", np.max(np.abs(weights[-1])),
        "Weights finite:", np.all(np.isfinite(weights[-1]))
    )


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
weights, biases, loss_lst = neural_network_multiple_hidden_layers([2,8,8,5], 2500, X_normalized, grade, 0.001, 0.9, 0.999, use_He=True)
mult_layer_accuracy, mult_confusion = test_my_neural_network(weights, biases, testing_X_normalized, testing_grade, 5)

print(mult_layer_accuracy)
print(mult_confusion)

plt.plot(loss_lst, marker='o')

plt.xlabel("index")
plt.ylabel("loss")

plt.show()






