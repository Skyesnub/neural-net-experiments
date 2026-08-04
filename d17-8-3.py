# DAY 17: VECTORIZING LINEAR REGRESSION

import numpy as np
from sklearn.linear_model import LinearRegression # wow im using skit learn!!!

hours_studied = np.random.randint(0,11,size=20)
hours_sleep = np.random.randint(0,11, size=20)
calories_eaten = np.random.randint(0,2000, size=20)
calories_eaten = calories_eaten / 200 # normalizing it so that it is between 0 and 10
base_points = np.random.randint(0,10) # this will be b for now

true_studied_m = 5
true_sleep_m = 4
true_calories_m = 1

grades = hours_studied * true_studied_m + hours_sleep * true_sleep_m + calories_eaten * true_calories_m + base_points
# making grades based on all the parameters

for i in range(len(grades)):
    grades[i] += np.random.randint(-5,6)
    pass
# adding some randomness to the dataset
    
print(f"Grades: {grades}")

def vectorized_linear_regression(X,y, learn_rate, loops):
    #X = np.column_stack((arr_1, arr_2, arr_3)) # combining them into one array, will make the next part a lot easier
    # now i am vectorizing outside the function
    m_count = X.shape[1] # doing the second dimension of X, which is the number of different ms there are
    n = X.shape[0] # first dimension of X, how many examples there are

    weights = np.zeros(m_count, dtype=float)
    b = 0 # starting values for each slope and b
    for i in range(loops):
        predicted = X @ weights + b # now using matrix multiplication!
        # there are for now three columns in weights and three columns in X? that's not how matrix multiplication works...
        # right?
        # after some reasoning its because its a 1d vector, not a 2d matrix, so in reality in this case
        # it will multiply each column by the respective column on weights
        losses = predicted - y     

        gradients = (2 / n) * X.T @ losses
        # we can also js use matrix multiplication for this!
        # this will create an array equal to the size / shape of weights because transposed x is a 3x20 array
        # and losses is a 20, array
        weights -= learn_rate * gradients
        # this becomes a lot simpler when everything is in one array

        mse = np.mean(losses ** 2)

        if i % (1000) == 0: # makes it so it only prints 100 times
            print(mse)

        b_gradient = (2 / len(X)) * np.sum(losses)

        b -= learn_rate * b_gradient

    return(weights, b)

learning_rate = 1e-4
outside_X = np.column_stack((hours_studied, hours_sleep, calories_eaten)) # column stacking outside the function this time

vectorized_lin_regression = vectorized_linear_regression(outside_X, grades, learning_rate, 100000)
print(f"Predicted weights (mine): {vectorized_lin_regression[0]}")
print(f"Predicted base score (mine): {vectorized_lin_regression[1]}")

model = LinearRegression()

model.fit(outside_X, grades)

print("Weights:", model.coef_)
print("Bias:", model.intercept_) # not entirely sure how this works yet, although coef returns a list, and intercept returns a b
# just like how linear regression works, and i'm also not entirely sure why you need to say model = LinearRegression()
# instead of just saying sklearn.linear_regression() or something
# sklearn is object oriented that's why

