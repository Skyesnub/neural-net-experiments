# DAY 4: STATISTICS

import numpy as np
import random

lower_bound = random.randint(20,60)
upper_bound = random.randint(90,110) # js so there is a higher likelihood of 100s

upper_bound = min(upper_bound, 100)

student_grades_list = np.random.randint(lower_bound,upper_bound, size=100)
print(f"Student grades: {student_grades_list}")

mean_score = student_grades_list.mean()
print(f"Mean of student scores: {mean_score}%")

standard_dev = student_grades_list.std()
print(f"Standard deviation of grades: {standard_dev}")
#don't normally use stdev, use cases:
# generally used to see how spread out a dataset is
# in finance, used to see how risky stocks are, how volatile things are
# etc etc...
# in this case, higher stdev means the scores were more spread out, 
# while lower stdev means they were more even

highest_grade_student = student_grades_list.argmax() # finds the index of the max
lowest_grade_student = student_grades_list.argmin() # finds the index of the min
#however, this only returns the first value like this, so there could be multiple.

print(f"Student {highest_grade_student} scored highest and student {lowest_grade_student} scored lowest.")



#way to show all students who scored highest / lowest, in case of multiple
highest_score = student_grades_list.max()
lowest_score = student_grades_list.min()

highest_students = np.where(student_grades_list == highest_score)[0]
lowest_students = np.where(student_grades_list == lowest_score)[0]

print(f"Student(s) {highest_students} scored {highest_score} and student(s) {lowest_students} scored {lowest_score}.")

median = np.median(student_grades_list) #also is used to find how spread out / not spread out
# the grades were, but in a diff way to stdev

print(f"Median score was {median}%")

variance = np.var(student_grades_list) # variance is sqrt of stdev, didn't know this before!
print(f"Variance was {variance}")

percentile = np.percentile(student_grades_list, 90) # percentile used to calculate what number is
# greater than x% of things in an array

print(f"90th percentile scored {percentile}")

unique_vals = np.unique(student_grades_list) #unsure why i would need this? js separates the unique
# values
num_unique_vals = len(unique_vals)

print(f"There are {num_unique_vals} unique grades here.")

#counting
num_students_above_avg = np.count_nonzero(student_grades_list > mean_score)
#will count how many times the statement parameter is true

print(f"{num_students_above_avg} students scored over the average score.\n")

a_scores = np.count_nonzero(student_grades_list >= 93)
b_scores = np.count_nonzero((93 > student_grades_list) & (student_grades_list >= 80))
c_scores = np.count_nonzero((80 > student_grades_list) & (student_grades_list >= 70))
d_scores = np.count_nonzero((70 > student_grades_list) & (student_grades_list > 60))
f_scores = np.count_nonzero(60 >= student_grades_list)
#so you can't do the regular python thing there, numpy throws a value error
#ValueError: The truth value of an array with more than one element is ambiguous.

print(f"{a_scores} students scored A's\n{b_scores} students scored B's\n{c_scores} students scored C's\n{d_scores} students scored D's\n{f_scores} students failed.\n")

big_grade_sheet = np.random.randint(0,101, (30,5))
print(f"Big grade sheet:\n {big_grade_sheet}")

each_average = np.mean(big_grade_sheet, axis=1)
print(f"Averages of each student: {each_average}")

best_student = np.argmax(each_average)
worst_student = np.argmin(each_average)

print(f"Worst student was {worst_student}, scoring {each_average[worst_student]} on average and best student was {best_student}, scoring {each_average[best_student]} on average.")

each_assignment_average = np.mean(big_grade_sheet, axis=0)

hardest_assignment = np.argmin(each_assignment_average)
easiest_assignment = np.argmax(each_assignment_average)

print(f"Hardest assignment was {hardest_assignment}, on average they scored {each_assignment_average[hardest_assignment]}.")
print(f"Easiest assignment was {easiest_assignment}, on average they scored {each_assignment_average[easiest_assignment]}.")



