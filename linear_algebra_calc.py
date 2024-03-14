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

class LinAlFuncs:

    def __init__(self, matrix, rows, cols):
        self.matrix = matrix
        self.rows = rows
        self.cols = cols
        
    # Identity Matrix
    def identity(self):
        id = np.identity(self.rows)
        print("Identity matrix:")
        print(id)

    # Find rank
    def rank(self):
        print("Rank of matrix:")
        

    # Find determinant
    def determinant(self):
        print("Determinant of matrix:")

    # Find inverse
    def inverse(self):
        if rows == columns:
            print("fart")
        print("Inverse of matrix:")

    # Find eigenvalues
    def eigenvalues(self):
        print("Eigenvalues of matrix:")

    # Find eigenvectors
    def eigenvectors(self):
        print("Eigenvectors of matrix:")

    # Find nullity
    def nullity(self):
        print("Nullity of matrix:")

    # Find dimension
    def dimension(self):
        print("Dimension of matrix:")

    # Find multiplicity
    def multiplicity(self):
        print("Multiplicity of matrix:")

    # Find linear (in)depenance
    def linear_independence(self):
        print("Linear indepenance of matrix:")

    # Find transpose
    def transpose(self):
        print("Transpose of matrix:")
        tp = np.transpose(self.matrix)
        print(tp)
        


f = LinAlFuncs(matrix, rows, columns)

while True:
    choice = input("What would you like to do with the matrix? (Type 'rank', 'det', 'inv', 'eigVal', 'eigVec', 'null', 'dim','mult', 'lindep', 'transp', 'all', or 'exit'): ")

    if choice == "rank":
        f.rank()
    elif choice == "det":
        f.determinant()  
    elif choice == "inv":
        f.inverse()
    elif choice == "eigVal":
        f.eigenvalues()
    elif choice == "eigVec":
        f.eigenvectors()
    elif choice == "null":
        f.nullity()
    elif choice == "dim":
        f.dimension()
    elif choice == "mult":
        f.multiplicity()
    elif choice == "lindep":
        f.linear_independence()
    elif choice == "transp":
        f.transpose()
    elif choice == "all":
        f.rank()
        f.determinant()
        f.inverse()
        f.eigenvalues()
        f.eigenvectors()
        f.nullity()
        f.dimension()
        f.multiplicity()
        f.linear_independence()
        f.transpose()
    elif choice == "exit":
        print("Exiting program.")
        break
    else:
        print("Invalid choice.")
        continue
