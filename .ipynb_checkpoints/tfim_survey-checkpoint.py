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
from scipy.optimize import curve_fit

# range of J_ij seeds
seed_range = range(10)

# define isWorking function
def isWorking(coeff_matrtix, perturbation_order, criterion = 0.5):
    isWorking_per_state = np.zeros(len(coeff_matrtix[:,1]))
    for i, par in enumerate(coeff_matrtix[:,1]):
        isWorking_per_state[i] = (abs(par - (perturbation_order+1)) <= criterion)
    return np.prod(isWorking_per_state), np.argwhere(isWorking_per_state == False)

# define power law fitting function
def power_law(x, A, b):
    return A*np.power(x, b)

# define analysis function
def tfim_analysis(L, Jij_seed, perturbation_order, h_x_range = np.arange(0, 0.005, 0.0001), PBC = True, J = 1):
    #Initialize the output dictionary containing all the information that we want to know about a specific instance
    #   - isEmpty
    #   - isWorking
    #   both of which contains logical True or False values

    #Initial set up
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
    GS_energy, GS_indices = tfim_perturbation.GS(Energies)

    # Specify perturbation order
    if perturbation_order == 3:
        analysis_func = tfim_perturbation.app_3_eigensystem_general_matrices
        H_app_3 = tfim_perturbation.H_app_3(basis, Jij, GS_indices, N, GS_energy)
        isEmpty = np.allclose(H_app_3, np.zeros((len(GS_indices), len(GS_indices))))
    elif perturbation_order == 4:
        analysis_func = tfim_perturbation.app_4_eigensystem_general_matrices
        isEmpty = False
    # Check to see if the max order perturbative term is empty and store this information in "info"
    info['isEmpty'] = isEmpty

    # Calculate approximated eigenvalues and eigenstates for range(h_x)
    app_eigenvalues, app_eigenstates = analysis_func(GS_indices, GS_energy, h_x_range, J, N,
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
    coeff_matrix = np.zeros((len(GS_indices), 2))
    for i in range(len(GS_indices)):
        pars, cov = curve_fit(f = power_law, xdata = h_x_range, ydata = error_array[i])
        coeff_matrix[i] = pars

    # Check to see if perturbation is working and store it in the info dictionary
    if info['isEmpty'] == False:
        judgment, error_classical_GS_index = isWorking(coeff_matrix, perturbation_order)
        info['isWorking'] = bool(judgment)
        info['error state index'] = error_classical_GS_index
        info['error order'] = coeff_matrix[error_classical_GS_index, 1]
    else:
        info['isWorking'] = None
        info['error order'] = None
        info['error state index'] = None

    # return info dictionary
    return info, coeff_matrix[:, 1]

# define survey function that uses 'analysis' to loop over a certain seed range
def survey(seed_range, number_of_spin, perturbation_order, printOrNot = False):
    # This function prints out all the info regarding each J_ij instance explicitly as well as returns arrays that store
    # this info
    print('Survey for 5 spin system: J_ij seed for range {}'.format(seed_range))
    if printOrNot:
        for i, seed in enumerate(seed_range):
            info, error_orders= tfim_analysis(number_of_spin, seed, perturbation_order)
            if info['isEmpty'] == True and info['isWorking'] != 1.:
                print('Error: seed {} does not have empty 3rd order matrix yet is not working.'.format(seed))
                print('The classical ground state causing the error is number {}'.format(info['error state index']))
                print('The error order is {}'.format(info['order']))
        return None
    else:
        info_arr = []
        isWorking_arr_num = np.zeros(len(seed_range))
        for i, seed in enumerate(seed_range):
            info, error_orders = tfim_analysis(number_of_spin, seed, perturbation_order)
            isWorking_arr_num[i] = info['isWorking']
            info_arr.append(info)
        # return an array of all the J_ij seeds for which the 3rd order works
        isWorking_arr = np.argwhere(isWorking_arr_num == 1.)[:, 0]
        return isWorking_arr, info_arr

def histogram(seed_range, number_of_spin, perturbation_order):
    err_order_hist = np.array([])
    for i, seed in enumerate(seed_range):
        info, error_orders = tfim_analysis(number_of_spin, seed, perturbation_order)
        if not info['isEmpty']:
            err_order_hist = np.append(err_order_hist, error_orders)
    # err_order_hist, bin_edges = np.histogram(err_order_hist, density=True)
    return err_order_hist
