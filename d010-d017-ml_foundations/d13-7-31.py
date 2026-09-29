# DAY 13: ACTUAL GRADIENT DESCENT

import numpy as np

learning_rate = 1e-8 # changed, 0.001 was way too large for this

sizes = np.random.randint(1000,3000, size=25) # copied the stuff from day 12

true_m = 1500
true_b = 100000

prices = sizes * true_m + true_b # making it so the data is way less random this time
prices += np.random.randint(-25000, 25000, size=25) # but obviously we still need some randomness

best_m = np.random.randint(500,2500) # again js decided to randomize because unsure what to do with the starting vals of these
best_b = np.random.randint(50000,150000)


def calculate_loss(m,b):
    each_price = sizes * m + b
    each_loss = prices - each_price

    return np.mean((each_loss)**2)

best_loss = calculate_loss(best_m, best_b) # set a baseline for the best loss
epsilon = 1e-4 # changed, 1e-5 was prob too small

for i in range(10000):
    m_gradient = ((calculate_loss(best_m-epsilon, best_b) # this is the equation for gradient descent
                 - calculate_loss(best_m+epsilon, best_b)) # had to search this up
                 / (2*epsilon))
    # if calculate_loss(best_m + epsilon, best_b) is larger, m_gradient should be negative, 1st is better
    # if calculate_loss(best_m + epsilon, best_b) is smaller, m_gradient should be positive, 2nd is better
    # this means that m_gradient is positive when the second number is better (you should add to m)
    # this means that when m_gradient is negative the first number is better (you should subtract from m)
    # this means that i should use a positive operator (-)

    best_m += learning_rate * m_gradient

    b_gradient = ((calculate_loss(best_m, best_b-epsilon) # this is the equation for gradient descent
                 - calculate_loss(best_m, best_b+epsilon)) # had to search this up
                 / (2*epsilon)) # basically asking if i move left and right, what is the slope btwn them

    best_b += learning_rate * b_gradient # again, probably could have used a helper function
    # so i didn't have to repeat the same code but i think it is fine
    # also, apparently i did this the negative way (the +epsilon goes first, not the -epsilon)
    # but it should still work

    best_loss = calculate_loss(best_m, best_b) # updating best loss

    if i % 100 == 0:
        print(best_loss)

print(f"My results: {best_m, best_b}")

polyfit_m, polyfit_b = np.polyfit(sizes, prices, 1) # just comparing to polyfit
predicted_polyfit_prices = sizes * polyfit_m + polyfit_b
polyfit_loss = np.mean((prices - predicted_polyfit_prices)**2)


print(f"Polyfit's results: {polyfit_m, polyfit_b}")
print(f"Polyfit has {polyfit_loss / best_loss * 100}% of my loss.")




