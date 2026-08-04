# DAY 7: NUMPY PROJECT

import numpy as np

PLAYER_NUM = 1000
STARTING_MONEY = 100
SPINS = 1000
# vars to make it easier for me

big_roulette_matrix = np.random.randint(0,37,size=(PLAYER_NUM, SPINS)) # initializing matrix for players
# and spins
money = np.full(PLAYER_NUM, STARTING_MONEY) # previously, was just gonna append a value to the end of
# big_roulette_matrix simulating their money, but that seemed like a weird thing to do

red_set = {1, 3, 5, 7, 9,
           12, 14, 16, 18,
           19, 21, 23, 25, 27,
           30, 32, 34, 36}
# if they always bet on red, technically they'd on avg be losing money, its 18/37 chance to win

print(big_roulette_matrix)

for player in range(len(big_roulette_matrix)): # started with a loop for now, unsure if there is a better way
    for spin in big_roulette_matrix[player]:
        if money[player] > 0: # only allowing them to play if they have money
            if spin in red_set: # right now assumes they always bet on red
                money[player] += 5
            else:
                money[player] -= 5

# there is a way to do this without a loop, but it would require allowing the player to bet no matter what
# first create a boolean matrix using np.isin
# then using np.where to make it so when the boolean matrix is true put 5 and when false put -5
# then add the np.where matrix back to the money matrix
# with this version, you can def notice its hicupping a little, especially when it gets bigger


print(f"Final money totals: {money}")

bankrupt_people = np.sum(money <= 0)

print(f"{bankrupt_people} out of {PLAYER_NUM} players went bankrupt.")

best_player = np.argmax(money)
best_player_money = money[best_player]
# took me a while, finally remembered argmax/min exists
print(f"Player {best_player} was the best with ${best_player_money} remaining.")

made_money_percent = (np.sum(money > STARTING_MONEY)) / PLAYER_NUM * 100
print(f"{made_money_percent}% of people made money")
