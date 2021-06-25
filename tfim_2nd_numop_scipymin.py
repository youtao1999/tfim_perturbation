import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize
import tfim
import tfim_perturbation
import matplotlib.pyplot as pl

# Initial system specification
L = [4]
Jij_seed = 19
h_x_range = np.linspace(0.001, 0.01, 1)

PBC = True
J = 1

# Build lattice and basis
###################################
lattice = tfim.Lattice(L, PBC)
N = lattice.N
basis = tfim.IsingBasis(lattice)
###################################

#construct random J matrix
Jij = tfim.Jij_instance(N,J,"bimodal",Jij_seed)

# List out all the spin_states, corresponding indices and energies
Energies = -tfim.JZZ_SK_ME(basis,Jij)
# for index in range(2**N):
#     print(index, basis.state(index), Energies[index])

#construct random J matrix
Jij = tfim.Jij_instance(N,J,"bimodal",Jij_seed)

# Build 2nd order approximated matrix

GS_energy, GS_indices = tfim_perturbation.GS(Energies)

H_app_0 = tfim_perturbation.H_app_0(GS_energy, GS_indices)

H_app_1 = tfim_perturbation.H_app_1(basis, GS_indices, N)

# Parametrize H_app_2

def H_app_2_param(basis, Jij, GS_indices, N, GS_energy, param):
    # Second-Order term in perturbation theory
    H_app_2 = np.zeros((len(GS_indices), len(GS_indices)))

    for column, GS_ket_1 in enumerate(GS_indices):
        state_1 = basis.state(GS_ket_1)
        for i in range(N):
            basis.flip(state_1, i)
            state_1_flipped_index = basis.index(state_1)
            if state_1_flipped_index not in GS_indices:
                energy_gap = tfim_perturbation.state_energy(basis, Jij, state_1_flipped_index) - GS_energy
                for j in range(N):
                    basis.flip(state_1, j)
                    ES_2_flipped_index = basis.index(state_1)
                    GS_2_index = np.argwhere(np.array(GS_indices) == ES_2_flipped_index)
                    if len(GS_2_index) > 0:
                        row = GS_2_index[0][0]
                        H_app_2[row, column] -= param / energy_gap
                    basis.flip(state_1, j)
            basis.flip(state_1, i)
    return H_app_2

# Build exact matrix
V_exc = tfim_perturbation.V_exact(basis, lattice)
H_0_exc = tfim_perturbation.H_0_exact(Energies)

# Define error function to be minimized
def err(x):
    param = 1.
    alpha = x
    # alpha is the fitting parameter for the 3rd order polynomial
    error_arr = np.zeros(np.shape(h_x_range))
    for i, h_x in enumerate(h_x_range):
        H_app = H_app_0 + h_x*H_app_1 + np.power(h_x, 2.)*H_app_2_param(basis, Jij, GS_indices, N, GS_energy, param)
        # Calculate the energy eigenvalue of the approximated 2nd order matrix
        app_eigenvalues, app_eigenstates = eigh(H_app)
        # print(app_eigenvalues)
        app_GS_eigenvalue = min(app_eigenvalues)
        # Calculate exact eigenvalues and eigenstates for range(h_x)
        exc_eigenvalues, exc_eigenstates = tfim_perturbation.exc_eigensystem(basis, h_x_range, lattice, Energies)
        # print(exc_eigenvalues)
        exc_GS_eigenvalue = min(exc_eigenvalues[:,i])
        error_arr[i] = abs(app_GS_eigenvalue-exc_GS_eigenvalue) - alpha*np.power(h_x, 3.)
    # print(error_arr)
    return np.sqrt(sum(np.power(error_arr, 2.)))

# Perform optimization
x_0 = 0.3
res = minimize(err, x_0, method = 'Nelder-Mead')
# print("optimized perturbation parameter: ", res.x[0], "optimized curve fitting parameter: ", res.x[1])
print(res.x)

# Fixed alpha and plot error function
#
# x_arr = np.zeros((10,2))
# x_arr[:,0] = np.linspace(0.5, 1.5, 10)
# x_arr[:,1] = 0.37*np.ones(10)
# err_arr = np.zeros(np.shape(x_arr[:,0]))
# for i in range(len(x_arr[:,0])):
#     err_arr[i] = err(x_arr[i])
# print(err_arr)
# fig = pl.figure(figsize=(8, 6))
# pl.rcParams['font.size'] = '18'
# pl.plot(x_arr[:,0],err_arr/len(err_arr), lw=1.3, ls='-', color="blue")
# pl.ylabel(r'$Error$', fontsize=18)
# pl.xlabel('Coefficient for 2nd order', fontsize=18)
# pl.xticks(fontsize=18)
# pl.yticks(fontsize=18)
# pl.tick_params('both', length=7, width=2, which='major')
# pl.tick_params('both', length=5, width=2, which='minor')
# pl.grid(True)
# pl.legend(loc=2, prop={'size': 16}, numpoints=1, scatterpoints=1, ncol=2)
# fig.tight_layout(pad=0.5)
# pl.savefig("Error_plot.png")
