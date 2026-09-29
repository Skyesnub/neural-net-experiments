# DAY 14: MSE AND MAE
# MSE and MAE are ways to calculate the loss of fitting algorithms
# AIs want to minimize the MSE and MAE to make sure their lines (or logic) are as fitted as possible

import numpy as np

sizes = np.random.randint(1000,3000, size=25) # copied the stuff from day 12

true_m = 1500
true_b = 100000

prices = sizes * true_m + true_b # making it so the data is way less random this time
prices += np.random.randint(-25000, 25000, size=25) # but obviously we still need some randomness

prices[0] += 500000 # adding a massive outlier, so that MSE will explode but MAE will be more or less fine

best_m = np.random.randint(500,2500) # again js decided to randomize because unsure what to do with the starting vals of these
best_b = np.random.randint(50000,150000)

polyfit_m, polyfit_b = np.polyfit(sizes, prices, 1) # fitting with polyfit first

print(polyfit_m, polyfit_b)

MSE_matrix = (prices - (polyfit_m * sizes + polyfit_b))**2 # making it a matrix first, then will find average
MSE = np.mean(MSE_matrix) # then find the mean to find the final MSE
print(f"MSE: {MSE}")
# MSE will heavily punish large outliers because it squares every error so some really large number would become so much larger
# by squaring it

MAE_matrix = np.abs(prices - (polyfit_m * sizes + polyfit_b)) # similar, but this time don't square it
MAE = np.mean(MAE_matrix) # instead, use absolute value
print(f"MAE: {MAE}")

# MAE makes me wonder, instead of using abs to find the absolute value, you could maybe just delete
# the abs and maybe in this scenario you are just trying to get the number as close to 0 as possible

ME_matrix = (prices - (polyfit_m * sizes + polyfit_b)) # ME stands for mean error (teehee)
ME = np.mean(ME_matrix) # similarly, it needs to be the mean
print(f"ME (Mean error): {ME}")
# plain old mean error can make a line that misses everything by a ton look completely accurate
# because the positives and negatives will cancel out. I thought this was a good thing because then the line
# would be in the middle of everything but that isn't really the case...