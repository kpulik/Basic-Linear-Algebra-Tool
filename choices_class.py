import numpy as np
from functions_class import Functions as funcs

class Choices(funcs):

    choice = input("What would you like to do with the matrix? (Type 'rank', 'det', 'inv', 'eigVal', 'eigVec', 'null', 'dim','mult', 'lindep', 'transp', 'all', or 'exit'): ")

    if choice == "rank":
        funcs.rank()
    elif choice == "det":
        funcs.det()
    elif choice == "inv":
        funcs.inv()
    elif choice == "eigVal":
        funcs.eigVal()
    elif choice == "eigVec":
        funcs.eigVec()
    elif choice == "null":
        funcs.null()
    elif choice == "dim":
        funcs.dim()
    elif choice == "mult":
        funcs.mult()
    elif choice == "lindep":
        funcs.lindep()
    elif choice == "transp":
        funcs.transp()
    elif choice == "all":
        funcs.rank()
        funcs.det()
        funcs.inv()
        funcs.eigVal()
        funcs.eigVec()
        funcs.null()
        funcs.dim()
        funcs.mult()
        funcs.lindep()
        funcs.transp()
    elif choice == "exit":
        print("Exiting program.")
    else:
        print("Invalid choice.")
        
    