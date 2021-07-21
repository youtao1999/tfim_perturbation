'''
    Tao You
    7/12/2021
    This code aims to search through a range of J_ij seeds to try to find which ones the 3rd order perturbation does
    work for and which ones it fails.
'''

'''
Work flow:
1. specify number of spins as well as the range of J_ij seeds to search through
2. For each instance of J_ij, we need certain information, including
    - first, check to see if the 3rd order matrix becomes 0 for this instance
    - second, check to see if the error has the correct order 
3. define function that returns the ordered error corresponding to 3rd order pertubation
    - this function first calculates all the eigenvalues and eigenstates for each h_x value for both the approximated
    Hamiltonian and exact Hamiltonian
    - then it calculates the error
    - then it interpolate the error to a polynomial and calculates the order of the error
    - finally it judges whether the 3rd order perturbation is working for this instance of J_ij based upon some
    arbitrarily specified threshold.
'''

import tfim
import tfim_perturbation
import numpy as np
from scipy import optimize

# range of J_ij seeds
seed_range = range(10)

# define isWorking function
def isWorking(coeff_matrtix, perturbation_order, criterion = 0.5):
    isWorking_per_state = np.zeros(len(coeff_matrtix))
    for i in range(len(coeff_matrtix)):
        coeff_arr = coeff_matrtix[i]
        norm = np.linalg.norm(coeff_arr)
        judging_criterion = 1 - coeff_arr[perturbation_order+1]/norm
        isWorking_per_state[i] = judging_criterion <= criterion
    return np.logical_and(isWorking_per_state)

# define analysis function
def tfim_analysis_3(L, Jij_seed, h_x_range = np.arange(0, 0.001, 0.00001), PBC = True, J = 1):

    #Initialize the output dictionary containing all the information that we want to know about a specific instance
    #   - isEmpty
    #   - isWorking
    #   both of which contains logical True or False values

    info = {}
    # Configure the number of spins to the correct format for analysis
    L = [L]

    # Build lattice and basis
    ###################################
    lattice = tfim.Lattice(L, PBC)
    N = lattice.N
    basis = tfim.IsingBasis(lattice)
    ###################################

    # construct random J matrix
    Jij = tfim.Jij_instance(N, J, "bimodal", Jij_seed)

    # List out all the spin_states, corresponding indices and energies
    Energies = -tfim.JZZ_SK_ME(basis, Jij)
    # for index in range(2 ** N):
    #     print(index, basis.state(index), Energies[index])

    # Build 3rd order approximated matrix

    GS_energy, GS_indices = tfim_perturbation.GS(Energies)

    H_app_0 = tfim_perturbation.H_app_0(GS_energy, GS_indices)

    H_app_1 = tfim_perturbation.H_app_1(basis, GS_indices, N)

    H_app_2 = tfim_perturbation.H_app_2(basis, Jij, GS_indices, N, GS_energy)

    H_app_3 = tfim_perturbation.H_app_3(basis, Jij, GS_indices, N, GS_energy)

    # Check to see if H_app_3 is empty and store this information in "info"
    info['isEmpty'] = np.allclose(H_app_3, np.zeros((len(GS_indices), len(GS_indices))))

    # Calculate approximated eigenvalues and eigenstates for range(h_x)
    app_eigenvalues, app_eigenstates = tfim_perturbation.app_3_eigensystem(GS_indices, GS_energy, h_x_range, J, N,
                                                                           basis, Jij)
    # Calculate exact eigenvalues and eigenstates for range(h_x)
    exc_eigenvalues, exc_eigenstates = tfim_perturbation.exc_eigensystem(basis, h_x_range, lattice, Energies)

    # Extract exact ground states
    exc_GS_eigenstates = np.zeros((len(h_x_range), len(GS_indices), len(GS_indices)))

    for i in range(len(h_x_range)):
        for m, j in enumerate(GS_indices):
            for n, k in enumerate(GS_indices):
                exc_GS_eigenstates[i, m, n] = exc_eigenstates[i, j, n]

    # Extract exact ground energy
    reordered_app_eigenstates = np.zeros([len(h_x_range), len(GS_indices), len(GS_indices)])
    epsilon = 1 * 10 ** (-6)

    for h_x_index in range(len(h_x_range)):
        if h_x_index < 2:
            reordered_app_eigenstates[h_x_index] = app_eigenstates[h_x_index]
        else:
            for k in range(len(GS_indices) // 2):
                fidelity_array = []
                for v1 in [reordered_app_eigenstates[h_x_index - 1, :, 2 * k],
                           reordered_app_eigenstates[h_x_index - 1, :, 2 * k + 1]]:
                    for v2 in [app_eigenstates[h_x_index, :, 2 * k], app_eigenstates[h_x_index, :, 2 * k + 1]]:
                        fidelity_array = np.append(fidelity_array, tfim_perturbation.fidelity(v1, v2))
                if abs(fidelity_array[0] - max(fidelity_array)) < epsilon:
                    reordered_app_eigenstates[h_x_index, :, 2 * k] = app_eigenstates[h_x_index, :, 2 * k]
                    reordered_app_eigenstates[h_x_index, :, 2 * k + 1] = app_eigenstates[h_x_index, :, 2 * k + 1]
                else:
                    reordered_app_eigenstates[h_x_index, :, 2 * k] = app_eigenstates[h_x_index, :, 2 * k + 1]
                    reordered_app_eigenstates[h_x_index, :, 2 * k + 1] = app_eigenstates[h_x_index, :, 2 * k]

    reordered_exc_GS_eigenstates = np.zeros([len(h_x_range), len(GS_indices), len(GS_indices)])
    epsilon = 1 * 10 ** (-12)

    for h_x_index in range(len(h_x_range)):
        if h_x_index < 2:
            reordered_exc_GS_eigenstates[h_x_index] = exc_GS_eigenstates[h_x_index]
        else:
            for k in range(len(GS_indices) // 2):
                fidelity_array = []
                for v1 in [reordered_exc_GS_eigenstates[h_x_index - 1, :, 2 * k],
                           reordered_exc_GS_eigenstates[h_x_index - 1, :, 2 * k + 1]]:
                    for v2 in [exc_GS_eigenstates[h_x_index, :, 2 * k], exc_GS_eigenstates[h_x_index, :, 2 * k + 1]]:
                        fidelity_array = np.append(fidelity_array, tfim_perturbation.fidelity(v1, v2))
                if abs(fidelity_array[0] - max(fidelity_array)) < epsilon:
                    reordered_exc_GS_eigenstates[h_x_index, :, 2 * k] = exc_GS_eigenstates[h_x_index, :, 2 * k]
                    reordered_exc_GS_eigenstates[h_x_index, :, 2 * k + 1] = exc_GS_eigenstates[h_x_index, :, 2 * k + 1]
                else:
                    reordered_exc_GS_eigenstates[h_x_index, :, 2 * k] = exc_GS_eigenstates[h_x_index, :, 2 * k + 1]
                    reordered_exc_GS_eigenstates[h_x_index, :, 2 * k + 1] = exc_GS_eigenstates[h_x_index, :, 2 * k]
    # Calculate and plot energy errors
    corrected_exc_eigenvalues = np.zeros((len(GS_indices), len(h_x_range)))

    for i in range(len(GS_indices)):
        for j in range(len(h_x_range)):
            corrected_exc_eigenvalues[i, j] = exc_eigenvalues[i, j]

    error_array = np.absolute(corrected_exc_eigenvalues - app_eigenvalues)

    # Curve fit
    coeff_matrix = np.zeros((5,len(GS_indices)))
    for i in range(len(GS_indices)):
        coeffs = np.polynomial.polynomial.polyfit(h_x_range, error_array[i], 4)
        coeff_matrix[:, i] = coeffs

    # Check to see if 3rd order perturbation is working and store it in the info dictionary
    info['isWorking'] = isWorking(coeff_matrix, perturbation_order = 3)

    # return info dictionary
    return info

info = tfim_analysis_3(5, 19)
print(info)