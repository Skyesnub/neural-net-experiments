# DAY 8: VECTORS

import numpy as np
import math

list_a = np.random.randint(1,10, size=5) #create small lists for dot product
list_b = np.random.randint(1,10, size=5)

print(f"List a: {list_a}")
print(f"List b: {list_b}")

dot_product_list = np.dot(list_a, list_b) #  <-- numpy's way of doing dot product

#making dot product myself
def dot_product(a,b):
    if len(a) != len(b):
        return 0 # i'm not sure if this is allowed for dot product, and it wouldn't make sense
                 # since i kinda know what dot product is and generally they need to be 
                 # the same size

    sum = 0 # initial var to add all to

    for i in range(len(a)):
        sum += (a[i] * b[i]) # having all my variables as single letters is eh...
    return sum # after loop is done

my_dot_product_list = dot_product(list_a, list_b)

print(f"Numpy's: {dot_product_list}\nMine: {my_dot_product_list}")

# dot product: useful in a lot of ways
# when it comes to vectors, the larger the number, the more the vectors point in the same direction or the larger
# their size
# aka, in machine learning, the more similar things are (and how long things are)
# however this can fail easily and thus is not the only metric to how similar things are
# also when comparing lists with larger numbers that are completely different it will be very big even if
# they're not similar
# cosine similarity does something similar to dot product, but it instead takes into account
# the direction of the vector so now you're really looking at things that are a lot similar.

def cosine_similarity(a,b):
    if len(a) != 2 or len(b) != 2:
        return 0 # not dealing with this

    dot = dot_product(a,b)

    mag_a = math.sqrt(a[0]**2 + a[1]**2) # previously just divided the ratios between a and b
    mag_b = math.sqrt(b[0]**2 + b[1]**2) # from each other but then realized that was kinda
                                         # really wrong
    # cosine similarity does actually take into account the dot product (which i thought it didnt)                        
    return dot / (mag_a * mag_b)


MINI = 0
MAXI = 10

vector_1 = np.random.randint(MINI,MAXI, size=2)
vector_2 = np.random.randint(MINI,MAXI, size=2)

#vector_1 = [9,5] #testing my theory of max dp being 200
#vector_2 = [9,5]

print(f"Vector 1: {vector_1}")
print(f"Vector 2: {vector_2}")

# min dot product should be 0 in this case (perpendicular), not including negatives
# max should be 162 2(9*9)

vector_dp = np.dot(vector_1, vector_2)

print(f"Vector dot product is: {vector_dp}") # not exactly thge best for telling similarity

cosine_sim = cosine_similarity(vector_1, vector_2) # better for telling similarity

print(f"Cosine similarity is: {cosine_sim}")

