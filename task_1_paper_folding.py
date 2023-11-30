import time

# Code to calculate the thickness when the paper is folded 43 times
THICKNESS = 0.00008

#[Problem 1] Implementation using exponentiation arithmetic operators
folded_thickness = THICKNESS * 2**43
print("Thickness: {} meters".format(folded_thickness))

#[Problem 2] Unit Conversion
print("Thickness: {:.2f} kilometers".format(folded_thickness / 1000))


#[Problem 3] Create using a for statement
THICKNESS = 0.00008
num_folds = 43

folded_thickness = THICKNESS

for _ in range(num_folds):
    folded_thickness *= 2

print("Thickness: {:.2f} kilometers".format(folded_thickness / 1000))

#[Problem 4] Comparison of calculation time

start = time.time()
folded_thickness = THICKNESS * 2**43
elapsed_time = time.time() - start
print("Using exponentiation: time {}[s]".format(elapsed_time))


start = time.time()
folded_thickness = THICKNESS
for _ in range(43):
    folded_thickness *= 2
elapsed_time = time.time() - start
print("Using for statement: time {}[s]".format(elapsed_time))

##[Problem 5] Saving to a list
THICKNESS = 0.00008
num_folds = 43

folded_thickness_list = [THICKNESS]

for _ in range(num_folds):
    folded_thickness *= 2
    folded_thickness_list.append(folded_thickness)

print("Number of elements in the list: {}".format(len(folded_thickness_list)))

#[Problem 6] Displaying a line graph
import matplotlib.pyplot as plt

# Display the graph
plt.title("Thickness of Folded Paper")
plt.xlabel("Number of Folds")
plt.ylabel("Thickness [m]")
plt.plot(folded_thickness_list)
plt.show()

#[Problem 7] Customizing graphs
plt.title("Thickness of Folded Paper")
plt.xlabel("Number of Folds")
plt.ylabel("Thickness [m]")
plt.plot(folded_thickness_list, color='green', linewidth=2, linestyle='--', marker='o', markersize=8)
plt.show()