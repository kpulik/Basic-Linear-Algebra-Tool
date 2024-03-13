import numpy as np
from linear_algebra_calc import matrix, rows

class Functions(matrix,rows):

    # Identity Matrix
    id = np.identity(rows)
    
    # Find rank
    def rank(matrix):
        #work on this
        print("Rank of matrix:")

    # Find determinant
    def det(matrix):
        #work on this
        print("Determinant of matrix:")

    # Find inverse
    def inverse(matrix):
        #work on this
        print("Inverse of matrix:")

    # Find eigenvalues
    def eigVal(matrix):
        #work on this
        print("Eigenvalues of matrix:")

    # Find eigenvectors
    def eigVec(matrix):
        #work on this
        print("Eigenvectors of matrix:")

    # Find nullity
    def null(matrix):
        #work on this
        print("Nullity of matrix:")

    # Find dimension
    def dim(matrix):
        #work on this
        print("Dimension of matrix:")

    # Find multiplicity
    def mult(matrix):
        #work on this
        print("Multiplicity of matrix:")

    # Find linear (in)depenance
    def lindep(matrix):
        #work on this
        print("Linear indepenance of matrix:")

    # Find transpose
    def transp(matrix):
        print("Transpose of matrix:")
        tp = np.transpose(matrix)
        print(tp)
        
        