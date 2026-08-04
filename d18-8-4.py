# DAY 18: LOGISTIC REGRESSION & BINARY CLASSIFICATION

import numpy as np

# linear regression is bad for classification because:
# it can return values below 0 and above 1, invalid probabilites
# uses MSE which is very sensitive to outliers
# forms a flat linear fit, not an s-curve, which is apparently very useful for transitioning between diff categories
# so instead logistic regression solves all those problems

# sigmoid function turns any number from a 0-1 in an s-curve graph shape

random_dataset = np.linspace(-10, 10, 21)

sigmoid_dataset = 1 / (1 + (np.exp(-random_dataset))) # sigmoid is NOT a normalization function, like what i thought
# it makes it so very negative numbers are almost 0 and very pos numbers are almost 1, 0 is 0.5

print(f"Sigmoid: {sigmoid_dataset}")

# probabilities are quite simple, they range from 0-1, 0 being impossible, 1 being certain

# BCE is quite simple conceptually
# basically the model will make predictions with probabilites, and then it measures how wrong it is using -log
# then it averages that value across all the data values

students = 100

hours_studied = np.random.randint(0, 10, size=students)
hours_sleep = np.random.randint(0, 10, size=students)

score = hours_studied * 10 + hours_sleep * 5
score += np.random.randint(-20,20,size=students) # now time for randomization (should make it wayy harder)

passed = score > 50
passed = passed.astype(int) # changes it to 0 or 1
print("Passed list:", passed)

X = np.column_stack((hours_studied, hours_sleep)) # doing X and y, similar to day 17
y = passed

def sigmoid(x): # generalizing sigmoid so i don't have to do that every single time
    return 1 / (1 + np.exp(-x))

def binary_cross_entropy(y, prediction):
    return -np.mean(
        y*np.log(prediction) +
        (1-y)*np.log(1-prediction)
    ) # not memeorizing yet

weights = np.zeros(X.shape[1])
b=0
n = X.shape[0] # how many students there are

learning_rate = 1e-2

for i in range(10000):
    prediction = X @ weights + b # same as last time!
    prediction = sigmoid(prediction) # turning it into sigmoid
 
    losses = prediction - y

    weight_gradient = (1/n) * X.T @ losses # again, this is written in a diff way than how i would normally
    # represent it but its fine
    bias_gradient = np.mean(losses)

    weights -= learning_rate * weight_gradient # updating weights and b
    b -= learning_rate * bias_gradient

    if i % 100 == 0:
        print(binary_cross_entropy(y, prediction))

prediction = (prediction >= 0.5).astype(int)
print("Predictions", prediction)

accuracy = np.mean(passed == prediction)

print(f"Accuracy: {accuracy * 100}%")

wanted_study_hours = int(input("How many hours would you want to study? "))
wanted_sleep_hours = int(input("How many hours would you want to sleep? "))

study_m = weights[0]
sleep_m = weights[1]

probability = (sigmoid(study_m * wanted_study_hours + sleep_m * wanted_sleep_hours + b))
print(f"We think your probability of passing is {probability*100}%.")


