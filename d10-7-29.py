# DAY 10: k-NN classifier

import numpy as np
import math

points = np.array([
    [1, 2],   # Red
    [2, 1],
    [2, 3],
    [3, 2],
    [3, 4],
    [4, 3],
    [4, 5],
    [5, 4],
    [5, 6],
    [6, 5],

    [4, 2],   # Blue
    [5, 1],
    [6, 2],
    [6, 3],
    [7, 3],
    [7, 4],
    [8, 4],
    [8, 5],
    [9, 5],
    [9, 6]
])

labels = np.array([
    "Red","Red","Red","Red","Red",
    "Red","Red","Red","Red","Red",

    "Blue","Blue","Blue","Blue","Blue",
    "Blue","Blue","Blue","Blue","Blue"
])

new_point = np.random.randint(1,10,size=2)

print(new_point)

#def get_distance(point_1, point_2):
    #horizontal_part = (point_1[0] - point_2[0]) ** 2 # probably could do this in js one line
    #vertical_part = (point_1[1] - point_2[1]) ** 2 # but for the sake of readability i did multiple

    #return math.sqrt(horizontal_part + vertical_part)
# nvm realized i could broadcast with a fully numpy solution

distances = np.sqrt(np.sum((points - new_point) ** 2, axis=1)) # broadcasts points across
# the entire distances array then does calculations for each individual element in the matrix

print(f"Distances: {distances}")

min_dist_index = np.argmin(distances)
label = labels[min_dist_index]

print(f"This point is most likely of label {label}, with 1 checker.") 

# now with three checkers

#generalizing
NUM_CHECKERS = 9

bottom_three_distances = np.argpartition(distances, NUM_CHECKERS)[:NUM_CHECKERS] # new numpy function, basically in this case
# it puts the element at index 3 in its place, puts smaller stuff below it and larger stuff above it
# need first three of the list because i only need the top three elements
# it returns the indices because (i just found this out) arg means argument meaning it returns indices

print(f"Bottom three distances indices: {bottom_three_distances}")

bottom_three_labels = labels[bottom_three_distances] # took me a bit to realize you can do this

print(bottom_three_labels) # yes i know these labels are stale after the generalization but its ok

red_count = np.sum(bottom_three_labels == "Red")
print(f"Reds: {red_count}") # yay sanity check
print(f"Blues: {NUM_CHECKERS - red_count}")



if red_count >= NUM_CHECKERS//2+1: # previously did NUM_CHECKERS - 1 but then realized that only worked for 3
    confidence = red_count / NUM_CHECKERS * 100
    print(f"This point is most likely of label Red, with {NUM_CHECKERS} checkers, with {confidence}% confidence.")
else:
    confidence = (NUM_CHECKERS-red_count) / NUM_CHECKERS * 100
    print(f"This point is most likely of label Blue, with {NUM_CHECKERS} checkers, with {confidence}% confidence.")