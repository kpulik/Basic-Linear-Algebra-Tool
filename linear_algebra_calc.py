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
print("Original matrix:\n")
print(matrix,"\n")

class LinAlFuncs:

    def __init__(self, matrix, rows, cols):
        self.matrix = matrix
        self.rows = rows
        self.cols = cols
        
    # Identity Matrix
    def identity(self):
        id = np.identity(self.rows)
        print("Identity matrix:\n")
        print(id,"\n")

    # Find rank
    def rank(self):
        rank = np.linalg.matrix_rank(self.matrix)

        if rank > min(self.rows, self.cols):
            print("Error: Rank cannot exceed matrix dimensions\n")
            return

        if np.all(self.matrix == 0):
            print("Rank is 0 (zero matrix)\n")
            return

        if self.rows == self.cols:
            if rank == self.rows:
                print("Matrix is full rank (invertible)\n")
            else:
                print("Matrix is not full rank (not invertible)\n")

        print("Rank of matrix:\n", rank,"\n")
        

    def determinant(self):
        matrix = self.matrix  # Use the matrix attribute of the class
        # Base case: if the matrix is a 2x2 matrix
        if len(matrix) == 2 and len(matrix[0]) == 2:
            return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]

        # Recursive case
        det = 0
        for c in range(len(matrix)):
            minor = np.concatenate((matrix[1:,:c], matrix[1:,c+1:]), axis=1) # Create minor
            minor_laf = LinAlFuncs(minor, len(minor), len(minor[0])) # Create new instance with minor
            det += ((-1)**c) * matrix[0][c] * minor_laf.determinant() # Call determinant on new instance
        print("Determinant of matrix:\n", det,"\n")



    # Find inverse
    def inverse(self):
        if rows == columns:
            inv = np.linalg.inv(self.matrix)
            print("Inverse of matrix:\n", inv)
        else:
            print("Inverse does not exist for non-square matrix\n")

    # Find eigenvalues
    def eigen(self):
        if rows == columns:
            # Calculate eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eig(self.matrix)

            print("Eigenvalues of matrix:\n")
            for eigenval in eigenvalues:
                print(f"{eigenval:.2f}","\n")

            print("\nEigenvectors of matrix:\n")
            for eigenvec in eigenvectors.T:
                print(eigenvec,"\n")

            # Check for zero eigenvalues
            zero_eigenvalues = np.isclose(eigenvalues, 0)
            if any(zero_eigenvalues):
                print("\nDirections that get squashed to zero (zero eigenvalues):\n")
                for eigenvec in eigenvectors.T[zero_eigenvalues]:
                    print(eigenvec,"\n")

            # Check for large eigenvalues
            large_eigenvalues = np.abs(eigenvalues) > 1
            if any(large_eigenvalues):
                print("\nDirections that get stretched or compressed (large eigenvalues):\n")
                for eigenval, eigenvec in zip(eigenvalues[large_eigenvalues], eigenvectors.T[large_eigenvalues]):
                    print(f"Eigenvalue: {eigenval:.2f}, Eigenvector: {eigenvec}","\n")
        else:
            print("Must be a square matrix!\n")
             
        

    # Find nullity
    def nullity(self):
        # Calculate the rank of the matrix
        rank = np.linalg.matrix_rank(self.matrix)
        
        # Calculate the nullity using the Rank-Nullity Theorem
        nullity = columns - rank
        
        print(f"Rank of the matrix: {rank}\n")
        print(f"Nullity of the matrix: {nullity}\n")
        
        if nullity == 0:
            print("The matrix has only the trivial solution (x = 0) for the homogeneous system Ax = 0.\n")
            print("This indicates that the rows/columns of the matrix are linearly independent (full rank).\n")
        else:
            print(f"The homogeneous system Ax = 0 has {nullity} free variables in its solution.\n")
            print("This signifies some level of redundancy or dependence among the rows/columns (less than full rank).\n")
        
        print(f"\nRank-Nullity Theorem: rank({rank}) + nullity({nullity}) = {columns} (number of columns in the matrix)\n")
        


    # Find multiplicity
    def multiplicity(self):
        if rows == columns:
            # Calculate eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eig(self.matrix)

            print("Algebraic Multiplicity of Eigenvalues:\n")
            unique_eigenvalues, counts = np.unique(eigenvalues, return_counts=True)
            for eigenval, count in zip(unique_eigenvalues, counts):
                print(f"Eigenvalue: {eigenval:.2f}, Algebraic Multiplicity: {count}\n")

            print("\nGeometric Multiplicity of Eigenvalues:\n")
            for eigenval in unique_eigenvalues:
                eigenvecs = eigenvectors[:, np.isclose(eigenvalues, eigenval)]
                geometric_multiplicity = np.linalg.matrix_rank(eigenvecs)
                print(f"Eigenvalue: {eigenval:.2f}, Geometric Multiplicity: {geometric_multiplicity}\n")

            print("\nRelationship Between Algebraic and Geometric Multiplicities:\n")
            for eigenval, count, geometric_multiplicity in zip(unique_eigenvalues, counts, [np.linalg.matrix_rank(eigenvecs) for eigenvecs in (eigenvectors[:, np.isclose(eigenvalues, eigenval)] for eigenval in unique_eigenvalues)]):
                if geometric_multiplicity == count:
                    print(f"For eigenvalue {eigenval:.2f}, the algebraic and geometric multiplicities are equal ({count}), indicating linearly independent eigenvectors.\n")
                else:
                    print(f"For eigenvalue {eigenval:.2f}, the algebraic multiplicity ({count}) is greater than the geometric multiplicity ({geometric_multiplicity}), suggesting some level of dependence among the eigenvectors.\n")
        else:
            print("Must be a square matrix!\n")

    # Find linear dependance
    def linear_independence(self):
        # Calculate the rank of the matrix
        rank = np.linalg.matrix_rank(self.matrix)
        
        # Check if the matrix is square
        if rows == columns:
            # For square matrices, check if rank is equal to the number of rows/columns
            if rank == rows:
                print("The matrix is linearly independent.\n")
            else:
                print("The matrix is linearly dependent.\n")
        else:
            # For non-square matrices, check if rank is equal to the smaller dimension
            if rank == min(rows, columns):
                print("The matrix is linearly independent.\n")
            else:
                print("The matrix is linearly dependent.\n")
        
        # Find the number of dependent columns/rows
        num_dependent = columns - rank
        if num_dependent > 0:
            print(f"There are {num_dependent} linearly dependent columns/rows in the matrix.\n")
        else:
            print("There are no linearly dependent columns/rows in the matrix.\n")
        
        # Find the linearly independent columns/rows
        independent_cols = np.nonzero(np.sum(np.abs(self.matrix), axis=0) != 0)[0]
        independent_rows = np.nonzero(np.sum(np.abs(self.matrix), axis=1) != 0)[0]
        
        print("\nLinearly independent columns:\n")
        print(independent_cols,"\n")
        
        print("\nLinearly independent rows:\n")
        print(independent_rows,"\n")


f = LinAlFuncs(matrix, rows, columns)

while True:
    choice = input("What would you like to do with the matrix? (Type 'rank', 'det', 'inv', 'eig', 'null', 'mult', 'lindep', 'all', or 'exit'): ")

    if choice == "rank":
        f.rank()
    elif choice == "det":
        f.determinant()  
    elif choice == "inv":
        f.inverse()
    elif choice == "eig":
        f.eigen()
    elif choice == "null":
        f.nullity()
    elif choice == "mult":
        f.multiplicity()
    elif choice == "lindep":
        f.linear_independence()
    elif choice == "all":
        f.rank()
        f.determinant()
        f.inverse()
        f.eigen()
        f.nullity()
        f.multiplicity()
        f.linear_independence()
    elif choice == "exit":
        print("Exiting program.\n")
        break
    else:
        print("Invalid choice.\n")
        continue
