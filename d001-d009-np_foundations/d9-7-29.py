# DAY 9: MORE MATRICES

import numpy as np

matrix_a = np.random.randint(1,11, size=(2,3))
matrix_b = np.random.randint(1,11, size=(3,4))

multiplied_matrix = matrix_a @ matrix_b

# remember, take the row from the first matrix and the column from the second matrix
# put them into a sort of list
# then find the dot product of all that and that will be the number in the result

print(f"{matrix_a}\n{matrix_b}")

print(f"Multipled:\n\n\n\n\n {multiplied_matrix}") # to hide answer until after

# multipled one matrix successfully first try, actually multiplying it by hand made me really
# realize why 2x3 @ 3x4 = 2x4 because the dot product lists are all gonna be length 3
# and there are only 2 rows to multiply by in the first matrix and 4 columns in the 2nd matrix
# generalizing: axb @ bxc = axc bc dot product lists are length b
# also matrix multiplication on its own is PAINFUL
# most work is done on hand, then checked with numpy