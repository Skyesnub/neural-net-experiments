# DAY 11: LINEAR REGRESSION MODEL

import numpy as np

sizes = np.array([1000, 1200, 1500, 1800, 2000]) # js copied the arrays again
prices = np.array([200000, 240000, 300000, 360000, 390000])

sizes = np.random.randint(1000,3000, size=25)
prices = np.random.randint(200000,400000, size=25)



predicted_prices = np.array(sizes * 0 + -1000) # putting in baseline numbers
error = np.array(predicted_prices - prices) # just so you don't have to put a constant like 1000000000
best_loss = np.mean(error ** 2) # or something

best_m = 0 # baseline bests if nothing better comes
best_b = -1000


for m in range(0,500): # from 0-500 with 1 increase, may change later
    for b in range(-1000, 1000):
        predicted_prices = np.array(sizes * m + b) # using cur m and b to predict prices
        error = np.array(predicted_prices - prices) # created a new error array using broadcasting!
        # i feel like its finally starting to come naturally
        # also you technically don't need np.array for the two lines above, but it feels
        # clearer to me
        new_loss = np.mean(error ** 2)

        if new_loss < best_loss: # if the new line is better than the old one
            best_loss = new_loss
            best_m = m # originbally named base_m and base_b and loss, but best for all of those names
            # prob fits better
            best_b = b

# polyfit is literally just a better and faster version of what i just wrote
# it returns a list of 2 numbers, and works for both linear and quadratic

polyfit_m, polyfit_b = np.polyfit(sizes, prices, 1)

print(f"My way: {best_m, best_b}")
print(f"With polyfit: {polyfit_m, polyfit_b}") # realizing that my b has a really small range for this
# but i don't want it to take a million years
# however because my b has such a low range polyfit and my answers are increasingly different

while True: # just to make it safer for me
    try:
        wanted_size = int(input("If you wanted a house, what size would it be? "))
        break
    except ValueError:
        print("no")

wanted_price = wanted_size * polyfit_m + polyfit_b
print(f"Since you wanted a {wanted_size} sqft house, it will cost about ${wanted_price}.")
        
