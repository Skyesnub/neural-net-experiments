# DAY 20: IMPROVING MULTICLASS CLASSIFICATION

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

true_score = X @ TRUE_WEIGHTS + TRUE_B + np.random.randint(-15,16,size=NUM_STUDENTS//10*8)
# will multiply the hours studied and slept by the weights, add b, then add a bit of randomness

grade = np.zeros(NUM_STUDENTS//10*8,dtype=int) # starting with 0s, to plan to change the values in the next few lines of code
# this time separated

testing_scores = testing_X @ TRUE_WEIGHTS + TRUE_B + np.random.randint(-15,16,size=NUM_STUDENTS//10*2) # these are the 200
# scores that the model will be testing

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

learning_rate = 0.1
y_onehot = np.eye(5)[grade]
n=X.shape[0]
b = np.zeros(5)

for i in range(10000):
    #logits = X @ weights + b # new thingy, logits, the weights are now a matrix and there is a logit for each grade (abcdf) and each class
    # slept and studied
    logits = X_normalized @ weights + b # normalziing!

    probabilities = softmax(logits) # softmax basically has all the probabilities add up to 1, kinda like percents

    loss = categorical_cross_entropy(
        grade,
        probabilities,
        is_sparse=True # just a new way of finding the loss, that needs the real values, the predicted probabilites
    )

    error = probabilities - y_onehot # basically the onehot thing puts a 1 wherever the actual grade is (wherever it is correct)
    # and probabilties has the predictions, so if there is a 90% chance wherever the 1 was (0.9 compared to 1) it would be less loss
    # and if there was a 10% chance where a 0 was it would also be less loss
    # howveer if it was a 0.1 where a 1 was then it would be a lot of loss

    weight_gradient = (1/n) * X_normalized.T @ error
    bias_gradient = (1/n) * np.sum(error, axis=0)

    weights -= learning_rate * weight_gradient
    b -= learning_rate * bias_gradient # rest is basically the same

    if i % 1000 == 0:
        print(loss)

print(weights, b)

testing_X_normalized = (testing_X - mean) / std

testing_grade[testing_scores >= 90] = 4   # A
testing_grade[(testing_scores >= 80) & (testing_scores < 90)] = 3   # B
testing_grade[(testing_scores >= 65) & (testing_scores < 80)] = 2   # C
testing_grade[(testing_scores >= 50) & (testing_scores < 65)] = 1   # D
testing_grade[testing_scores < 50] = 0    # F

#logits = testing_X @ weights + b # now using the testing values
logits = testing_X_normalized @ weights + b # now using the testing normalized

testing_probabilites = softmax(logits)

testing_predictions = np.argmax(testing_probabilites, axis=1)
accuracy = np.mean(testing_predictions == testing_grade)

print("Accuracy:", accuracy) # for troubleshooting

confusion_matrix = np.zeros((5,5),dtype=int)

for actual,predicted in zip(testing_grade, testing_predictions): # this loop will just add a 1 to wherever the predictions and actual
    confusion_matrix[actual,predicted] += 1 # grade was
# f should be on top a should be on bottom

print(confusion_matrix)



