import numpy as np
from tfim_lanczos import lanczos
import os
import shutil

seed_range = np.arange(1, 100, 1)
h_x_range = np.linspace(0., 2., 20)
PBC = True
h_z = 0.001
maxiter = 400
L = [2,2]

exc_eigenvalues_all = np.zeros((len(seed_range), len(h_x_range)))
first_excited__exc_energies_all = np.zeros((len(seed_range), len(h_x_range)))
second_derivative_exc_eigenvalues_all = np.zeros((len(seed_range), len(h_x_range)))
chi_arr_all = np.zeros((len(seed_range), len(h_x_range)))
S_SG_arr_all= np.zeros((len(seed_range), len(h_x_range)))
for seed in seed_range:
    N, h_x_range, exc_eigenvalues, first_excited__exc_energies, second_derivative_exc_eigenvalues, chi_arr, S_SG_arr = lanczos(L, seed, h_x_range, PBC, h_z, maxiter)
    exc_eigenvalues_all[seed] = exc_eigenvalues
    first_excited__exc_energies_all[seed] = first_excited__exc_energies
    second_derivative_exc_eigenvalues_all[seed] = second_derivative_exc_eigenvalues
    chi_arr_all[seed] = chi_arr
    S_SG_arr_all[seed] = S_SG_arr

exc_eigenvalues_ave = np.mean(exc_eigenvalues_all, axis = 0)
first_excited__exc_energies_ave = np.mean(first_excited__exc_energies_all, axis = 0)
second_derivative_exc_eigenvalues_ave = np.mean(second_derivative_exc_eigenvalues_all, axis = 0)
chi_arr_ave = np.mean(chi_arr_all, axis = 0)
S_SG_arr_ave = np.mean(S_SG_arr_all, axis = 0)

# output files
# check to see whether the output file already exists
if os.path.isdir("lanczos_ave_data"):
    shutil.rmtree("lanczos_ave_data")

# make output directory
output = "lanczos_ave_data"
os.mkdir(output)
os.chdir(output)

outF = open("lanczos_ave data_{size} seed from {init} to {final}.txt".format(size = L, init = seed_range[1], final = seed_range[-1]), 'w')
for i, h_x in enumerate(h_x_range):
    outF.write("{index} {h_x_val} {GS_energy_val} {first_excited_energy_val} {second_derivative_energy_val} {susceptibility_val} {structure_factor_val} \n".format(index = i, h_x_val = h_x, GS_energy_val = exc_eigenvalues[i], first_excited_energy_val = first_excited__exc_energies[i], second_derivative_energy_val = second_derivative_exc_eigenvalues[i], susceptibility_val = chi_arr[i], structure_factor_val = S_SG_arr[i]))
    # outF.write("{index} {h_x_val} {GS_energy_val} {first_excited_energy_val} {second_derivative_energy_val} {structure_factor_val} \n".format(index = i, h_x_val = h_x, GS_energy_val = exc_eigenvalues[i], first_excited_energy_val = first_excited__exc_energies[i], second_derivative_energy_val = second_derivative_exc_eigenvalues[i], structure_factor_val = S_SG_arr[i]))
outF.close()