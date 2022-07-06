import tfim
import tfim_perturbation
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla
import matplotlib.pyplot as pl
import matplotlib.ticker as mtick
import time

# Initial system specification
L = [4, 4]
init = 0.001
final = 4.
num_steps = 20
h_x_range = np.linspace(init, final, num_steps)
PBC = True
J = 1
p = 0.5

# Build lattice and basis
lattice = tfim.Lattice(L, PBC)
N = lattice.N
basis = tfim.IsingBasis(lattice)

# modified function to eigendecompose the exact Hamiltonian using Lanczos method
def exc_eigensystem(basis, h_x_range, lattice, Energies):
    # Calculate exact eigenvalues and eigenstates for range(h_x)
    exc_eigenvalues = np.zeros(len(h_x_range))
    first_excited_exc_energies = np.zeros(len(h_x_range))
    exc_eigenstates = np.zeros((len(h_x_range), basis.M))
    V_exc_csr = tfim_perturbation.V_exact_csr(basis, lattice)
    H_0_exc_csr = tfim_perturbation.H_0_exact_csr(Energies)
    for j, h_x in enumerate(h_x_range):
        H = H_0_exc_csr - V_exc_csr.multiply(h_x)
        exc_eigenvalue, exc_eigenstate = spla.eigsh(H, k=2, which="SA", v0=v0, maxiter=400, return_eigenvectors=True)
        exc_eigenvalues[j] = exc_eigenvalue[0]
        first_excited_exc_energies[j] = exc_eigenvalue[1]
        for k in range(basis.M):
            exc_eigenstates[j][k] = exc_eigenstate[k, 0]
    return V_exc_csr, H_0_exc_csr, exc_eigenvalues, first_excited_exc_energies, exc_eigenstates

num_iter = 1
chi_arr_all = np.zeros((num_iter, len(h_x_range)))
order_param_all = np.zeros((num_iter, len(h_x_range)))
for i in range(num_iter):
    seed = int(time.time() * 1000) % num_iter ** 3
    # initialize Lanczos vector
    v0 = np.zeros(2 ** N)
    for i in GS_indices:
        v0[i] = 1
    print(seed)
    Jij = tfim_perturbation.Jij_2D_NN(seed, N, PBC, L[0], L[1], lattice)
    Energies = -tfim.JZZ_SK_ME(basis, Jij)
    GS_energy, GS_indices = tfim_perturbation.GS(Energies)
    V_exc, H_0_exc, exc_eigenvalues, first_excited__exc_energies, exc_eigenstates = exc_eigensystem(basis, h_x_range, lattice, Energies)
    chi_arr, order_param_arr = tfim_perturbation.susceptibility(h_x_range, lattice, basis, exc_eigenvalues, H_0_exc, V_exc, v0, h_z = 0.001)
    chi_arr_all[i] = chi_arr
    order_param_all[i] = order_param_arr

chi_arr_ave = np.mean(chi_arr_all, axis=0)
order_param_ave = np.mean(order_param_all, axis=0)

outF = open("lanczos_susceptibility_ave_{size}.txt".format(size = L), 'w')
for i, h_x in enumerate(h_x_range):
    outF.write("{index} {h_x_val} {chi_arr_val} {order_param_val} \n".format(index = i, h_x_val = h_x,  chi_arr_val = chi_arr_ave[i], order_param_val = order_param_ave[i]))
    # outF.write("{index} {h_x_val} {GS_energy_val} {first_excited_energy_val} {second_derivative_energy_val} {structure_factor_val} \n".format(index = i, h_x_val = h_x, GS_energy_val = exc_eigenvalues[i], first_excited_energy_val = first_excited__exc_energies[i], second_derivative_energy_val = second_derivative_exc_eigenvalues[i], structure_factor_val = S_SG_arr[i]))
outF.close()

# Susceptibility plot
fig = pl.figure(figsize=(8, 6))
pl.rcParams['font.size'] = '18'
pl.plot(h_x_range, chi_arr_ave, lw=1.3, ls='-', color="blue", label="susceptibility")
# pl.plot(h_x_range, lattice.N/h_x_range, lw = 1.3, ls='-', color="red", label= "lattice.N/h_x")
pl.ylabel(r'$\chi_{SG}$', fontsize=22)
pl.xlabel(r'$h_x/J_0$', fontsize=22)
pl.xticks(fontsize=18)
pl.yticks(fontsize=18)
pl.tick_params('both', length=7, width=2, which='major')
pl.tick_params('both', length=5, width=2, which='minor')
pl.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.6f'))
pl.grid(False)
pl.yscale('log')
# pl.ylim((0, 100))
pl.legend(loc=0, prop={'size': 16}, numpoints=1, scatterpoints=1, ncol=1)
fig.tight_layout(pad=0.5)

# Order parameter plot
fig = pl.figure(figsize=(8, 6))
pl.rcParams['font.size'] = '18'
pl.plot(h_x_range, order_param_ave / lattice.N, lw=1.3, ls='-', color="blue")
pl.ylabel(r'$<S_i>}$', fontsize=22)
pl.xlabel(r'$h_x/J_0$', fontsize=22)
pl.xticks(fontsize=18)
pl.yticks(fontsize=18)
pl.tick_params('both', length=7, width=2, which='major')
pl.tick_params('both', length=5, width=2, which='minor')
pl.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.6f'))
pl.grid(False)
# pl.yscale('log')
# pl.ylim((0, 1000))
# pl.legend(loc=0, prop={'size': 16}, numpoints=1, scatterpoints=1, ncol=1)
fig.tight_layout(pad=0.5)