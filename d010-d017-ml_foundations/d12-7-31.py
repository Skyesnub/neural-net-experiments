# DAY 12: GRADIENT DESCENT (its actually more like hill climbing but yeah)
# dw i didn't delete day 11 i just made a new project

import numpy as np

sizes = np.random.randint(1000,3000, size=25)

true_m = 1500
true_b = 100000

prices = sizes * true_m + true_b # making it so the data is way less random this time
prices += np.random.randint(-25000, 25000, size=25) # but obviously we still need some randomness

best_m = np.random.randint(500,2500) # unsure how i was supposed to make a basline so js made it random
best_b = np.random.randint(0,200000)

learning_rate = 5 # unsure if this is really big, also more called a step size

for i in range(100000):
    learning_rate = max(learning_rate * 0.999, 0.01) # making the learning rate smaller each time
    # hopefully should make it so it doesnt keep flip flopping back and forth over and over
    m1 = best_m - learning_rate
    m2 = best_m + learning_rate

    predicted_m1_prices = sizes * m1 + best_b # predicting the price when just m changes, m1 slotted in
    predicted_m2_prices = sizes * m2 + best_b # predicting the price when just m changes, m2 slotted in

    m1_loss = np.mean((prices - predicted_m1_prices)**2) # calculating the loss for both m1 and 2
    m2_loss = np.mean((prices - predicted_m2_prices)**2)

    if m1_loss > m2_loss: # m2 is better
        best_m = m2
    else: # m1 is better, or they're equal (which i would consider unlikely)
        best_m = m1

    b1 = best_b - learning_rate # doing the exact same thing all over again for b
    b2 = best_b + learning_rate # maybe could make a function for to make code look better
    # bc im doing it twice, but i think it's fine

    predicted_b1_prices = sizes * best_m + b1 # doing the same thing from earlier, but for b
    predicted_b2_prices = sizes * best_m + b2

    b1_loss = np.mean((prices - predicted_b1_prices)**2)
    b2_loss = np.mean((prices - predicted_b2_prices)**2)

    if b1_loss > b2_loss: # b2 is better
        best_b = b2
    else: # b1 is better, or they're equal (which i would consider unlikely)
        best_b = b1

    current_predictions = sizes * best_m + best_b
    loss = np.mean((prices - current_predictions) ** 2)

    #if i % 1000 == 0:
        #print(i, loss) # seeing how the loss develops over time
    # this stuff is commented out because printing takes too much time

polyfit_m, polyfit_b = np.polyfit(sizes, prices, 1) # again testing against polyfit
predicted_polyfit_prices = sizes * polyfit_m + polyfit_b
polyfit_loss = np.mean((prices - predicted_polyfit_prices)**2)

print("My way:", best_m, best_b, "My loss:", loss)
print("With polyfit:", polyfit_m, polyfit_b, "Polyfit's loss:", polyfit_loss)
print(f"Polyfit has {polyfit_loss / loss * 100}% of my loss.")


