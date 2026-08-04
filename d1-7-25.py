# DAY 1: ARRAYS

import numpy as np

oned_array = np.array([1,2,3])
#creates a one dimensional array with numbers 1,2,3

print(oned_array, "\n")

two_d_array = np.array([[1,2],[3,4]])
#creates and 2d array with number 1,2 then 3,4

print(two_d_array, "\n")

filled_array = np.full((2,2), 5)
#creates a full array completely filled with a constant number (this case 5)
# in a certain area (in this case 2x2)

print(filled_array, "\n")

zero_array = np.zeros((2,5), dtype=int)
#creates an array full of 0s in a certain set of dimensions (in this case 2x5)

print(zero_array, "\n")

ones_array = np.ones((2,5), dtype=int)
#creates an array full of 1s in a certain set of dimensions (this case 2x5)

print(ones_array, "\n")

ranged_array = np.arange(10) 
#will output an array of [0,1,2,3,4,5,6,7,8,9], 0 - (param-1)

print(ranged_array, "\n")

spaces_ranged_array = np.arange(1, 27, 3)
#will output an array that starts at 1, goes to 27-1, in increments of 3 (in this case).

print(spaces_ranged_array, "\n")

linspaced_array = np.linspace(0,100,26, dtype=int)
#outputs an array putting 26 numbers spaced evenly between 0 and 100
#(would normally use 25 but this includes both 0 and 100 so its 26)

print(linspaced_array, "\n")

checkerboard_array = np.zeros((8,8), dtype=int)
#starting with 0s

checkerboard_array[1::2, ::2] = 1 #changes even rows, odd columns to 1
#1::2 is 1 to the end for the rows spaced by 2, ::2 is 0 to the end spaced by 2
checkerboard_array[::2, 1::2] = 1 #changes odd rows, even columns to 1
#1::2 is 1 to the end for the rows spaced by 2, ::2 is 0 to the end spaced by 2

print(checkerboard_array)

massive_random_matrix = np.random.randn(100,100)
#creates a matrix of 100x100 that has random numbers in it
#standard normal distribution (mean = 0, standard deviation = 1).

np.set_printoptions(threshold=np.inf) #this line of code makes it so that it actually prints the whole matrix
print(massive_random_matrix)

np.set_printoptions() #resets it back so that the amount of stuff it prints can be limited again

print(massive_random_matrix[:5, :5])   # first 5 rows and first 5 columns

print(massive_random_matrix[0]) # first row
print(massive_random_matrix[:, 3]) #4th column
#the : in the 1st part of tuple means all rows, 3 means just the 4th (3rd) element

print(massive_random_matrix[2,5]) #prints the element in 3rd row, 6th column

#to remember for future, in numpy, it is array[row, column], and i can use : to mean all

shape = massive_random_matrix.shape #finds the shape, returns a tuple of the shape
size = massive_random_matrix.size #finds the number of elements in the matrix, 
                                    #returns an integer of how many elements
dtype = massive_random_matrix.dtype #returns the datatype of the elements in the matrix

print("Shape:", shape)
print("Size:", size)
print("Data type:", dtype)