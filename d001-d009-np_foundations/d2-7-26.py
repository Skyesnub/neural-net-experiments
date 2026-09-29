# DAY 2: INDEXING AND SLICING

import numpy as np

base_array = np.random.randint(-10, 10, size=5) #creates an array of random ints from -10 to 10 (not inclusive)
                                                # of size 5 (this is an archaic function but is ok)

base_2d_array = np.random.randint(-10, 10, size=(5,5)) #does the exact same as before but 5x5 (2d)

print(f"Base: {base_array}")
print(f"Base 2d: {base_2d_array}")

reversed_array = base_array.copy()
#create a copy to start :D
#also since when do you need to use .copy() i've never used it once and never had problems
#with it and apparently its true for js too my main language

for i in range(len(reversed_array)//2):
    reversed_array[i], reversed_array[-i-1] = reversed_array[-i-1], reversed_array[i] #reversing!!
    #the -1 in (-i-1) is because it starts at 0, and for going into the negatives of lists 
    #negative 0 isn't really a thing
    #in 2d matrices this will only reverse the rows
    #this is the manual way

print(f"Reversed: {reversed_array}")

also_reversed_array = reversed_array[::-1]

print(f"Better reversed array: {also_reversed_array}")
#son... forgot you could do this... reversed_array = reversed_array[::-1]

alternating_extracted_rows = []

for i in range(len(base_2d_array)): #there's prob a better way to do this
    if i % 2 == 0:
        alternating_extracted_rows.append(base_2d_array[i].tolist()) # adds that row
        #previously thought that you could js use += like w/ python lists, but you can't
        # with numpy, it generates an error
        # also .tolist() makes it so it doesn't say array([1,2,3,4,5]) and instead js
        # [1,2,3,4,5]

print(f"Every other row extracted: {alternating_extracted_rows}")

negative_to_zero = base_array.copy()

for i in range(len(negative_to_zero)):
    if negative_to_zero[i] < 0:
        negative_to_zero[i] = 0
    #simple

print(f"Negatives turned into 0s: {negative_to_zero}")

def swap_elements(row_1, row_2, in_list): #probably a much more efficient way but its ok
    returning_list = []
    
    for i in range(len(in_list)):
        if row_1 == i:
            returning_list.append(int(in_list[row_2]))
        elif row_2 == i:
            returning_list.append(int(in_list[row_1]))
        else:
            returning_list.append(int(in_list[i]))

        #using int here changes it from being annoying and saying
        #np.int64()
        #also += didn't work here, unsure why?


    return returning_list

swapped_list = base_array.copy()

print(f"Rows 0 and 3 swapped: {swap_elements(0,4,swapped_list)}")

evens_list = []

for i in base_array:
    if i % 2 == 0:
        evens_list.append(int(i))
        #again int is so that it wouldn't be annoying with np.int64
        #also += didn't work here, maybe i should js use append consistently

print(f"Only evens: {evens_list}")

#NOTES:

#generally, don't typecast to ints unless its for pretty printing :)
#remember that this exists: reversed_array = reversed_array[::-1]
#numpy is cool it can do the following instead of looping through the whole thing:
#negative_to_zero[negative_to_zero < 0] = 0
#base_array[base_array % 2 == 0]



