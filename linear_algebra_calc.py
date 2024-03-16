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

    # Constructor
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
        

    # Find determinant
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
        
    
    # Addition
    def addition(self):
        num_matrices = int(input("Enter the number of matrices to add: "))
        
        for _ in range(num_matrices):
            addition_matrix = np.zeros((self.rows, self.cols))  # Initialize a matrix to add
            
            # Populate the addition matrix with user input
            for i in range(self.rows):
                for j in range(self.cols):
                    addition_matrix[i][j] = int(input("Enter element (number) at row " + str(i) + " and column " + str(j) + ": "))
            
            print("Matrix being added:\n")
            print(addition_matrix, "\n")  # Print the current addition matrix
            
            if addition_matrix.shape == self.matrix.shape:
                self.matrix += addition_matrix
            else:
                print("Cannot add matrices with different dimensions.")

        print("Updated matrix after addition of all input matrices:\n")
        print(self.matrix, "\n")
        
        
    # Subtraction
    def subtraction(self):
        num_matrices = int(input("Enter the number of matrices to subtract: "))
        
        for _ in range(num_matrices):
            subtraction_matrix = np.zeros((self.rows, self.cols))  # Initialize a matrix to subtract
            
            # Populate the subtraction matrix with user input
            for i in range(self.rows):
                for j in range(self.cols):
                    subtraction_matrix[i][j] = int(input("Enter element (number) at row " + str(i) + " and column " + str(j) + ": "))
            
            print("Matrix being subtracted:\n")
            print(subtraction_matrix, "\n")  # Print the current subtraction matrix
            
            if subtraction_matrix.shape == self.matrix.shape:
                self.matrix -= subtraction_matrix
            else:
                print("Cannot subtract matrices with different dimensions.")

        print("Updated matrix after subtraction of all input matrices:\n")
        print(self.matrix, "\n")
    
    
    # Multiplication
    def multiplication(self):
        num_matrices = int(input("Enter the number of matrices to multiply: "))

        for _ in range(num_matrices):
            multiplication_matrix = np.zeros((self.rows, self.cols))  # Initialize a matrix to multiply

            # Populate the multiplication matrix with user input
            for i in range(self.rows):
                for j in range(self.cols):
                    multiplication_matrix[i][j] = int(input("Enter element (number) at row " + str(i) + " and column " + str(j) + ": "))

            print("Matrix being multiplied:\n")
            print(multiplication_matrix, "\n")  # Print the current multiplication matrix

            if multiplication_matrix.shape[1] == self.matrix.shape[0]:
                # Multiply the matrices using np.dot()
                self.matrix = np.dot(self.matrix, multiplication_matrix)
            else:
                print("Number of rows in the multiplication matrix must be equal to the number of columns in the original matrix for multiplication.")
        
        print("Updated matrix after multiplication with all input matrices:\n")
        print(self.matrix, "\n")

    
    # Division
    def division(self):
        num_matrices = int(input("Enter the number of matrices to divide: "))

        for _ in range(num_matrices):
            division_matrix = np.zeros((self.rows, self.cols))  # Initialize a matrix to divide

            # Populate the division matrix with user input
            for i in range(self.rows):
                for j in range(self.cols):
                    division_matrix[i][j] = int(input("Enter element (number) at row " + str(i) + " and column " + str(j) + ": "))

            print("Matrix to divide by:\n")
            print(division_matrix, "\n")  # Print the current division matrix

            if division_matrix.shape[0] == division_matrix.shape[1]:
                # Calculate the inverse of the division matrix
                inverse_matrix = np.linalg.inv(division_matrix)

                if inverse_matrix.shape[0] == self.matrix.shape[1]:  # Check compatibility for matrix division
                    # Perform matrix division by multiplying the original matrix with the inverse matrix
                    self.matrix = np.dot(self.matrix, inverse_matrix)
                else:
                    print("Number of columns in the division matrix must be equal to the number of rows in the original matrix for division.")
            else:
                print("Division matrix must be square for matrix division.")

        print("Updated matrix after division with all input matrices:\n")
        print(self.matrix, "\n")


    # Diagonalization
    def diagonalization(self):
        if self.rows == self.cols:
            eigenvalues, eigenvectors = np.linalg.eig(self.matrix)
            D = np.diag(eigenvalues)  # Construct the diagonal matrix D
            P = eigenvectors  # Eigenvectors as columns to form matrix P    
            P_inv = np.linalg.inv(P)  # Calculate the inverse of matrix P    
            diagonalized_matrix = np.dot(np.dot(P, D), P_inv)  # Calculate A = PDP^-1   
            print("Matrix A:\n")
            print(self.matrix, "\n")    
            print("Matrix D:\n")
            print(D, "\n")    
            print("Matrix P:\n")
            print(P, "\n")    
            print("Matrix P^-1:\n")
            print(P_inv, "\n")    
            print("Diagonalized matrix A = PDP^-1:\n")
            print(diagonalized_matrix, "\n")
        else:
            print("Matrix must be square for diagonalization.\n")



f = LinAlFuncs(matrix, rows, columns)

while True:
    choice = input("What would you like to do with the matrix? (Type 'rank', 'det', 'inv', 'eig', 'null', 'multiplicity', 'lindep', 'add', 'sub', 'mult', 'div', 'diag', or 'exit'): ")

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
    elif choice == "multiplicity":
        f.multiplicity()
    elif choice == "lindep":
        f.linear_independence()
    elif choice == "add":
        f.addition()
    elif choice == 'sub':
        f.subtraction()
    elif choice == 'mult':
        f.multiplication()
    elif choice == 'div':
        f.division()
    elif choice == 'diag':
        f.diagonalization()
    elif choice == "exit":
        print("Exiting program.\n")
        break
    else:
        print("Invalid choice.\n")
        continue
