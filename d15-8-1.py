# DAY 15: LINEAR REGRESSION PROJECT

import numpy as np

hours_studied = np.random.randint(0,25, size=100) # multiple features now!
hours_sleep = np.random.randint(2,11, size=100)
calories_eaten = np.random.randint(0,2000, size=100)

true_studied_m = 2
true_sleep_m = 3
true_calories_m = 0.005
true_b = 10

scores = (hours_studied * true_studied_m + 
          hours_sleep * true_sleep_m +
          calories_eaten * true_calories_m +
          true_b)

scores += np.random.randint(-5, 6, size=100) # adding a bit of randomness
print(f"Actual scores:\n {scores}")

studied_m = np.random.randint(1,3)
sleep_m = np.random.randint(1,5)
calories_m = (np.random.randint(1,10)/1000) # already noticing a problem
# i probably should be normalizing the data so that this is better because the learning rate
# will probably be too high
b = np.random.randint(0,20)



def calculate_loss(studied, sleep, calories,local_b):
    each_predicted_score = hours_studied * studied + hours_sleep * sleep + calories_eaten * calories + local_b
    each_loss = scores - each_predicted_score

    return np.mean((each_loss)**2)

loss = calculate_loss(studied_m, sleep_m, calories_m, b)

epsilon = 1e-4
learning_rate = 1e-9

for i in range(10000):
    studied_m_gradient = ((calculate_loss(studied_m+epsilon, sleep_m, calories_m, b) - 
                          calculate_loss(studied_m-epsilon, sleep_m, calories_m, b)) / 
                          (2*epsilon))
    # gradient descent thingy from day 13
    studied_m -= studied_m_gradient * learning_rate

    # same thing for sleep? (could again do a helper function but im kinda not wanting to do that)
    sleep_m_gradient = ((calculate_loss(studied_m, sleep_m+epsilon, calories_m, b) - 
                          calculate_loss(studied_m, sleep_m-epsilon, calories_m, b)) / 
                          (2*epsilon))
    sleep_m -= sleep_m_gradient * learning_rate
    # same thing for calories? (could again do a helper function but im kinda not wanting to do that)
    # im also gonna divide it by like 1000 more since calories is a bigger number, meaning that calories
    # m is gonna be way smaller.
    # this could also be solved with normalization
    calories_m_gradient = ((calculate_loss(studied_m, sleep_m, calories_m+epsilon, b) - 
                          calculate_loss(studied_m, sleep_m, calories_m-epsilon, b)) / 
                          (2*epsilon))
    calories_m -= calories_m_gradient * learning_rate / 1000
    # then finally the same thing for b
    b_gradient = ((calculate_loss(studied_m, sleep_m, calories_m, b+epsilon) - 
                 calculate_loss(studied_m, sleep_m, calories_m, b-epsilon)) / 
                 (2*epsilon))
    b -= b_gradient * learning_rate

    if i % 100 == 0:
        loss = calculate_loss(studied_m, sleep_m, calories_m, b)
        print(f"Loss: {loss}")

print(studied_m, sleep_m, calories_m, b)
print(f"For this test, hours studied matters with multipler {studied_m/true_studied_m},\n sleep matters with multipler {sleep_m/true_sleep_m}, \n, and calories matters with multipler {calories_m/true_calories_m}.")

wanted_study = int(input("If you took a test, how many hours would you want to study?")) # sanity check
wanted_sleep = int(input("If you took a test, how many hours would you want to sleep?"))
wanted_calories = int(input("If you took a test, how many calories would you want to eat?"))

predicted_score = (wanted_study * studied_m) + (wanted_sleep * sleep_m) + (wanted_calories * calories_m) + b

print(f"Your predicted score would be {predicted_score}%.") # this is according to my gradient descent
# so its not perfect