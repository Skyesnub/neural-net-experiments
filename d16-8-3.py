# DAY 16: ANALYTICAL GRADIENTS

import numpy as np

sizes = np.random.randint(500,2000, size=20)

TRUE_M = 500
TRUE_B = 100000

prices = sizes * TRUE_M + TRUE_B

for i in range(len(prices)):
    prices[i] += np.random.randint(-10000, 10000) # randomizing a little bit

m = 0 # 0 is what most people will use for linear regression
b = 0

learning_rate = 1e-7

for i in range(100000):
    predicted = sizes * m + b # current prediction given cur values

    losses = prices - predicted # apparently i did this the negative way again

    m_gradient = (2/len(sizes)) * np.sum(losses * sizes)
    # explanation cuz confused
    # (losses * sizes) is because you're taking all the losses, then multiplying them by the sizes
    # because larger sizes will be affected more by a change in m
    # then you add all those losses together
    # basically for every example, measure how wrong it is, then weigh it by how much it will affect the slope
    # then average them
    # its kinda written in a diff way than i would have written it, which is why it confused me.
    # i would have said 2 * np.sum(losses * sizes) / len(sizes)

    m += m_gradient * learning_rate
    # because i did this the negative way, i would use += here
    # if i did the positive way (predicted - prices) then i would use -=

    b_gradient = (2/len(sizes)) * np.sum(losses)
    # the reason this is different from the other one is because b affects everything the same
    # so there's no reason to multiply everything by sizes because there's not that difference

    b += b_gradient * learning_rate

    mse = np.mean(losses**2) # calculating the mse with the same thing from earlier
    if i % 10000 == 0:
        print(f"MSE: {mse}")

# generalizing
def analytical_gradient(arr_1, arr_2, learn_rate, loops):
    if len(arr_1) != len(arr_2):
        raise ValueError("Length of arrays do not match.")

    x_min = arr_1.min()
    x_max = arr_1.max()

    arr_1 = (arr_1 - x_min) / (x_max - x_min)
    #arr_2 = (arr_2 - arr_2.min()) / (arr_2.max() - arr_2.min()) not normalizing 2nd one for now

    m = 0
    b = 0 # base values
    for _ in range(loops):
        predicted = arr_1 * m + b
        losses = predicted - arr_2
        # swiched to positive version for readability

        m_gradient = (2/len(arr_1)) * np.sum(losses * arr_1)
        # same things as earlier
        m -= m_gradient * learn_rate

        b_gradient = (2/len(arr_1)) * np.sum(losses)

        b -= b_gradient * learn_rate

    # return what the final m and b should hopefully be, after unnormalizing
    actual_m = m / (x_max - x_min)
    actual_b = b - (m * x_min)/(x_max-x_min)
    return actual_m,actual_b


print(f"Equation: y = {m}x + {b}")
print(f"With generalized function: {analytical_gradient(sizes, prices, 1e-3, 100000)}")

