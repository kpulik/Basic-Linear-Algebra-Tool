import numpy as np
from choices_class import Choices as choices

# Get matrix dimensions from user
rows = int(input("Enter number of rows (m): "))
columns = int(input("Enter number of columns (n): "))

# Initialize matrix of zeros 
matrix = np.zeros((rows, columns))

# Populate matrix with user input
for i in range(rows):
    for j in range(columns):
        matrix[i][j] = int(input("Enter element at row " + str(i) + " and column " + str(j) + ": "))
        
# Print original matrix
print("Original matrix:")
print(matrix)

choices(matrix)