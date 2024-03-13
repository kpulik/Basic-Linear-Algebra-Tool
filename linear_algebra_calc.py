import numpy as np

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

# Define Identity Matrix
id = np.identity(rows)
print("Identity matrix:")
print(id)

# SPLIT THIS INTO MUTIPLE FILES/CLASSES AND USE CS CONCEPTS TO CREATE THIS PROGRAM

# Find rank
#work on this

# Find determinant
#work on this

# Find inverse
#work on this

# Find eigenvalues
#work on this

# Find eigenvectors
#work on this

# Find nullity
#work on this

# Find dimension
#work on this

# Find multiplicity
#work on this

# Find linear (in)depenance
#work on this

# Find transpose
print("Transpose of matrix:")
print(matrix.T)