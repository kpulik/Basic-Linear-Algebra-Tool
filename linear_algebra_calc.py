import numpy as np

# Get matrix dimensions from user
rows = int(input("Enter number of rows (m): "))
columns = int(input("Enter number of columns (n): "))

# Initialize matrix of zeros 
matrix = np.zeros((rows, columns))

# Populate matrix with user input
for i in range(rows):
    for j in range(columns):
        matrix[i][j] = int(input("Enter element (number) at row " + str(i) + " and column " + str(j) + ": "))
        
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
        rank = np.linalg.matrix_rank(self.matrix)

        if rank > min(self.rows, self.cols):
            print("Error: Rank cannot exceed matrix dimensions")
            return

        if np.all(self.matrix == 0):
            print("Rank is 0 (zero matrix)")
            return

        if self.rows == self.cols:
            if rank == self.rows:
                print("Matrix is full rank (invertible)")
            else:
                print("Matrix is not full rank (not invertible)")

        print("Rank of matrix:", rank)


    def determinant(self, matrix):

        m = self.rows 
        n = self.cols
        
        # Check if matrix is square
        if m != n:
            raise ValueError("Matrix must be square")

        if n == 1:
            return matrix[0,0]

        if n == 2:
            # Formula for 2x2 determinant
            return matrix[0,0]*matrix[1,1] - matrix[0,1]*matrix[1,0]

        # Recursive cofactor expansion for nxn matrix
        det = 0
        for col in range(n):
            minor = np.delete(matrix, (col), axis=1) 
            minor = np.delete(minor, (0), axis=0)
            cofactor = (-1)**(col+2) * matrix[0, col]  
            det += cofactor * self.determinant(minor)

        print("Determinant of matrix:", det)


    # Find inverse
    def inverse(self):
        if rows == columns:
            inv = np.linalg.qr(self.matrix)
            print("Inverse of matrix:", inv)
        else:
            print("Inverse does not exist for non-square matrix")

    # Find eigenvalues
    def eigenvalues(self):
        eigs = np.linalg.eigvals(self.matrix)
        print("Eigenvalues of matrix:", eigs)
             

    # Find eigenvectors
    def eigenvectors(self):
        eigvecs = np.linalg.eig(self.matrix)
        print("Eigenvectors of matrix:", eigvecs)
        

    # Find nullity
    def nullity(self):
        print("Nullity of matrix:", np.linalg.matrix_rank(self.matrix, tol=None) - self.rows)
        

    # Find dimension
    def dimension(self):
        print("Dimension of matrix:", self.rows, "x", self.cols)

    # Find multiplicity
    def multiplicity(self):
        eigs = np.linalg.eigvals(self.matrix)
        print("Multiplicity of eigenvalues:")
        print(np.bincount(np.round(eigs).astype(int)))

    # Find linear (in)depenance
    def linear_independence(self):
        print("Linear (In)Dependance of matrix:", np.linalg.matrix_rank(self.matrix) == min(self.rows, self.cols))

    # Find transpose
    def transpose(self):
        print("Transpose of matrix:", np.transpose(self.matrix))
        


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
