# DAY 5: MATRICES

import numpy as np

matrix_a = np.random.randint(0,10, size=(2,2)) #creating two matrices of same size
matrix_b = np.random.randint(0,10, size=(2,2))

print(f"Matrix A:\n {matrix_a}")
print(f"Matrix B:\n {matrix_b}")

new_matrix = matrix_a @ matrix_b
reg_multipled_matrix = matrix_a * matrix_b

print(f"Multiplied together using @:\n {new_matrix}")
#long long long explanation
# For each element in the result matrix, take one row from the first matrix
# and one column from the second matrix.
# example:
# [1,2] and [10,20]
# [3,4]     [30,40]
# for top left: row = [1,2], column = [10,30]
# then you multiply the first numbers together and multiply the second numbers together and add
# them up.
# (1*10) + (2*30) = 70

# there are other examples of how to use this like in a bakery where you have foods
# and how much of diff ingredients you need for them 
# one matrix for amount ingredients needed for each food over multiple days
# and one matrix for amount of each food wanted over multiple days
# then you know how much to buy via matrix multiplication

print(f"Multiplied together using *:\n {reg_multipled_matrix}") # in this case, each value 
# is simply multipled together by their corresponding value (assuming same exact size and shape)
# which we have seen earlier

matrix_c = np.random.randint(0,100,size=(3,2))

print(f"Matrix C:\n {matrix_c}")

transposed_matrix = matrix_c.T
print(f"Matrix C transposed:\n {transposed_matrix}")
# i can think of it either way, where first row becomes first column, second row becomes second column, etc...
# or i can also think of it vice versa, both work, which is why double transposing reverts back
# to original

double_transposed_matrix = transposed_matrix.T
print(f"Matrix C double transposed:\n {double_transposed_matrix}")
# for some reason i thought it wasn't reversible like that but transposing the matrix twice does actually
# change it back to its original i think

identity_matrix = np.eye(2)
# creates a 2x2 identity matrix
# an identity matrix is a square matrix where everything is zeros except for 1's
# when x and y index are the same
# during matrix multiplication, you can multiply a matrix by an identity matrix and it will be the exact same
# as it was before

print(f"Identity matrix:\n {identity_matrix}")

print(f"Matrix A before identity multiplication:\n {matrix_a}")
identity_multipled_a = matrix_a @ identity_matrix
print(f"Matrix A after identity multiplication:\n {identity_multipled_a}")
# it's the exact same as before!

matrix_power_matrix = np.linalg.matrix_power(matrix_a, 2) # will do A @ A. 3 will do A @ A @ A
# different from doing matrix_a ** 2, which is elementwise
matrix_elementwise_power = matrix_a ** 2

print(f"The entire matrix to the power of 2:\n {matrix_power_matrix}")
#proving they are different, but they're both probably useful!
print(f"Each element to the power of 2 (different):\n {matrix_elementwise_power}")

#when it comes to matrix multiplication and dimensions: this is what it should look like:
# (m × n) @ (n × p) = (m × p)
# where m n and p are all dimensions