# DAY 6: RANDOM

import numpy as np

dice_rolls = np.random.randint(1,7,size=100000) #making a lot of dicerolls
#using random (which i've already used)

#analyzing data:
num_ones = np.count_nonzero(dice_rolls==1) #count how many times this is true
#another common (probably more common) way of doing this is by saying
#np.sum(dice_rolls==1)
num_twos = np.sum(dice_rolls==2)
num_threes = np.sum(dice_rolls==3)
num_fours = np.sum(dice_rolls==4)
num_fives = np.sum(dice_rolls==5)
num_sixes = np.sum(dice_rolls==6)
#same for all the others

print(f"Number of 1's: {num_ones}")
print(f"Number of 2's: {num_twos}")
print(f"Number of 3's: {num_threes}")
print(f"Number of 4's: {num_fours}")
print(f"Number of 5's: {num_fives}")
print(f"Number of 6's: {num_sixes}")

# can also do it like this:
# counts = np.bincount(rolls)[:1]
# will count the number of rolls of each dice, not including 0 because zero is not a roll
# format: array [9,12,3,65,32,5] (really bad example but that's the idea)
# basically bincount builds a table where at 1 index, there is the frequency of the value that
# corresponds to that index (very useful!)

average_roll = np.mean(dice_rolls)
print(f"Average of rolls: {average_roll}")

lucky_dice = average_roll >= 3.5
margin = average_roll - 3.5

print(f"It is {lucky_dice} that you are luckier than average. Margin: {margin}") 
#lol idk why i put this here

coin_flips = np.random.randint(1,3,size=100000)

num_heads = np.sum(coin_flips==1)
num_tails = np.sum(coin_flips==2)

print(f"There were {num_heads} heads and {num_tails} tails.")

if num_heads > num_tails:
    print(f"There were {num_heads-num_tails} more heads than tails.")
elif num_tails > num_heads:
    print(f"There were {num_tails-num_heads} more tails than heads.")
else:
    print(f"Wow they're equal!")
#unsure what else to analyze but there is def going to be some more random stuff in the future

die_1_rolls = np.random.randint(1,7,size=10000)
die_2_rolls = np.random.randint(1,7,size=10000)

sum_dies = die_1_rolls + die_2_rolls
doubles = np.count_nonzero(die_1_rolls == die_2_rolls) # you can actually compare between multiple
# matrices using count_nonzero (and for np.sum as well, I tested it)
good_rolls = np.sum((die_1_rolls + die_2_rolls) > 10)

probabilites = np.bincount(sum_dies)[2:]

print(f"Probabilites from getting each sum from 2-12: {probabilites}")
print(f"Amount of doubles: {doubles}")
print(f"You rolled a really good roll (11 or 12) {good_rolls} times. ")


birthdays_matrix = np.random.randint(1,366,size=30)

unique_birthdays_matrix = np.unique(birthdays_matrix) # removes all non-unique things, useful to
# see if repeats
has_duplicates = unique_birthdays_matrix.size != birthdays_matrix.size # then compare them
num_duplicates = birthdays_matrix.size - unique_birthdays_matrix.size
# and find if there are duplicates
# honestly thought there would be a numpy function for that but i guess not

print(f"There were {num_duplicates} duplicate birthdays.")

big_birthdays_matrix = np.random.randint(1,366,size=(1000,25)) #sim 1000 diff groups
# i accidentally did this the other way around, like (30,1000)
# meaning i was simulating 30 groups of 1000 diff people

duplicate_count = 0

for i in big_birthdays_matrix:
    unique_bday_matrix = np.unique(i) # removes all non-unique things
    has_dupe = unique_bday_matrix.size != i.size
    if has_dupe:
        duplicate_count += 1
# theres probably? a way to do this without loops but i think? this is fine
# just researched, there is a way to do this without loops
# first you sort every group of 30 people
# then use np.any to see if there is anything to left and right of each value
# on axis = 1
# this will return the sum of the duplicates
# much more efficient but in my opinion my code is clearer (not necessarily better tho)

print(f"In 1000 sims, {duplicate_count} had duplicate birthdays")
