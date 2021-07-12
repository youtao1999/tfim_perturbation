'''
    Tao You
    7/12/2021
    This code aims to search through a range of J_ij seeds to try to find which ones the 3rd order perturbation does
    work for and which ones it fails.
'''

'''
Work flow:
1. specify number of spins as well as the range of J_ij seeds to search through
2. define function that returns the ordered error corresponding to 3rd order pertubation
    - this function first calculates all the eigenvalues and eigenstates for each h_x value for both the approximated
    Hamiltonian and exact Hamiltonian
    - then it calculates the error
    - then it interpolate the error to a polynomial and calculates the order of the error
    - finally it judges whether the 3rd order perturbation is working for this instance of J_ij based upon some
    arbitrarily specified threshold.
'''