# DAY 3: BROADCASTING

import numpy as np

random_number_matrix = np.random.randint(-10,10,size=(5,5))

print(f"Origial matrix:\n {random_number_matrix}")

random_number_matrix += [1,0,0,0,0] #this will only add 1 to the first column

print(f"Added 1 to first column:\n {random_number_matrix}")
#you can also js use slicing but it's cool
random_number_matrix[:, 0] += 1

print(f"Added another 1 to first column:\n {random_number_matrix}")

# normalization very important!! done with broadcasting!
# most common is min max normalization (formula: x-min/max-min)
# x is the whole array so every element in the list is changed by (-min/max-min)

random_number_list = np.random.randint(-10,10, size = 10)

print(f"A random list: {random_number_list}")

maxi = random_number_list.max()
mini = random_number_list.min()

normalized_list = (random_number_list - mini)/(maxi - mini)

print(f"A normalized list: {normalized_list}")
#this code also works for matrices just as well

weather_list = np.random.randint(32, 100, size = 10)
print(f"Some perfectly normal weather in Fahrenheit: {weather_list}")

celsius_weather_list = (weather_list-32)/9*5
#subtracts 32 from everything, divides it by 9, the multiplies by 5 using broadcasting

print(f"Weather in Celsius: {celsius_weather_list}")

#using random_number_matrix from earlier
print(f"Random matrix: {random_number_matrix}")

changing_each_column = random_number_matrix * [2,3,4,5,6]
#multiples 1st column by 2, 2nd by 3, etc...
# this works becuase it treats it like this:
# [2 3 4 5 6]
# [2 3 4 5 6]
# [2 3 4 5 6]
# [2 3 4 5 6]
# [2 3 4 5 6]

print(f"Each number multiplied by [2,3,4,5,6]: {changing_each_column}")

print(f"OG matrix (again): {random_number_matrix}")

rows_multipled = random_number_matrix * [1,2,3,4,5]

columns_multipled = random_number_matrix * [[1],[2],[3],[4],[5]] #you have to go outside the smaller list
                                                                 #to get to next row
                                                                 #(idk if i got these backwards)

print(f"Each row multipled by 1,2,3,4,5: {rows_multipled}")
print(f"Each column multipled by 1,2,3,4,5: {columns_multipled}")

#normalizing each columm
print(f"Random number matrix: {random_number_matrix}")

mini = random_number_matrix.min(axis=0) # the axis=0 makes them output lists, which is why
#the broadcasting works. Example: [[1,-1,3]
                                 #[4,5,6]]
#mini would return [-1,4]
maxi = random_number_matrix.max(axis=0)

normalized_columns_matrix = (random_number_matrix-mini)/(maxi-mini) #then the lists
#of maxi and mini then get broadcasted across random_number_matrix

print(f"Normalized each column: {normalized_columns_matrix}")

new_matrix = np.random.randint(0,10,size=(5,5))

print(f"New random matrix:\n {new_matrix}")

average_rows_val = new_matrix.mean(axis=1, keepdims=True) #axis=1 gets the mean from each row
#the keepdims is so that it keeps the same dimensions as before so that it does each ROW,
#not each COLUMN
average_rows_subtracted_matrix = new_matrix-average_rows_val #then broadcasts it across list like earlier

print(f"Subtracted by average of their row: {average_rows_subtracted_matrix}")

average_columns_val = new_matrix.mean(axis=0)
average_columns_subtracted_matrix = new_matrix-average_columns_val

print(f"Subtracted by average of their columns: {average_columns_subtracted_matrix}")