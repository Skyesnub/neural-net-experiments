import numpy as np

list_a = np.random.randn(6,2) #the 1 and 2 are dimensions, first param is vert, 2nd is hori
list_b = np.random.randn(2,6)

#print(list_a, list_b,"\n")
print(list_a, "\n")
print(list_b, "\n")

def compute_mean(list):
    reps = 0
    sum = 0

    for i in list:
        for j in i:
            sum += j
            reps += 1

    return sum / reps

def normalize_matrix(list):
    return (list - list.min()) / (list.max() - list.min())


#print(normalize_matrix(list_b))
print(list_a @ list_b)