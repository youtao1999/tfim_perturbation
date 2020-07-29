#!/usr/bin/env python

""""TFIMED.py
    Tao You
    07/02/2020
    --Build first and second order approximated matrices using perturbation theory
    --Requires: numpy, scipy.sparse, scipy.linalg, progressbar
"""
import tfim
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla
from scipy import linalg
import matplotlib.pyplot as plt
from scipy import optimize
import progressbar
import argparse
import os

###############################################################################

# To be used in tfim_search:

def Hamming_distance(state_1, state_2):
    # Define Hamming distance
    return len(np.nonzero(state_1 - state_2)[0])

def Hamming_array(GS_indices, basis):
    # Calculate Hamming distance array
    Hamming = np.zeros((len(GS_indices), len(GS_indices)-1))
    for i, n in enumerate(GS_indices):
        basis_n = basis.state(n)
        for j, m in enumerate(np.delete(GS_indices, i)):
            basis_m = basis.state(m)
            Hamming[i, j] = Hamming_distance(basis_n, basis_m)
    return Hamming

def judge(order, array):
    # See if minimum Hamming distance is two (order = 2)
    return np.amin(array) == order

###############################################################################

def state_energy(basis,J,state_index):
    """ Computes specific state energy"""
    
    shift_state = np.zeros(basis.N,dtype=int)
    state = basis.spin_state(state_index)
    energy = 0
    for shift in range(1,basis.N/2+1):
        shift_state[shift:] = state[:-shift]
        shift_state[:shift] = state[-shift:]
        if (basis.N%2 == 0) and (shift == basis.N/2):
            energy = energy + 0.5*np.dot(J[shift-1,:]*shift_state,state)
        else:
            energy = energy + np.dot(J[shift-1,:]*shift_state,state)
    energy = energy*(-1)
    return energy

def GS(Energies):
    GS_energy = np.min(Energies)
    GS_indices = np.nonzero(Energies == GS_energy)[0]
    return GS_energy, GS_indices
    
def H_app_0(GS_energy, GS_indices):
    return GS_energy*np.identity(len(GS_indices))

def H_app_1(basis, GS_indices, N):
    
    # First-Order term in perturbation theory
    V = np.zeros((len(GS_indices), len(GS_indices)))
    
    for column, ket in enumerate(GS_indices):
        state = basis.state(ket)
        for i in range(N):
            basis.flip(state,i)
            bra = basis.index(state)
            subspace_index = np.argwhere(GS_indices == bra)
            if len(subspace_index) > 0:
                row = subspace_index[0][0]
                V[row, column] += 1
            basis.flip(state,i)
    return V

def H_app_2(basis, Jij, GS_indices, N, GS_energy):
    # Second-Order term in perturbation theory
    H_app_2 = np.zeros((len(GS_indices), len(GS_indices)))
    
    for column, GS_ket_1 in enumerate(GS_indices):
        state_1 = basis.state(GS_ket_1)
        for i in range(N):
            basis.flip(state_1, i)
            state_1_flipped_index = basis.index(state_1)
            if state_1_flipped_index not in GS_indices:
                energy_gap = state_energy(basis, Jij, state_1_flipped_index) - GS_energy
                for j in range(N):
                    basis.flip(state_1,j)
                    ES_2_flipped_index = basis.index(state_1)
                    GS_2_index = np.argwhere(np.array(GS_indices) == ES_2_flipped_index)
                    if len(GS_2_index) > 0:
                        row = GS_2_index[0][0]
                        H_app_2[row, column] -= 1/energy_gap
                    basis.flip(state_1, j)
            basis.flip(state_1, i)
    return H_app_2

def H_app(h_x, H_0, V, H_2, J):
    # Calculate final approximated matrix
    c = h_x**2/J
    return H_0 - h_x*V + H_2*c

def V_exact(basis, lattice):
    V_exact = np.zeros((basis.M, basis.M))
    for ket in range(basis.M):
        state = basis.state(ket)
        for i in range(lattice.N):
            basis.flip(state,i)
            bra = basis.index(state)
            V_exact[bra, ket] += 1
            basis.flip(state,i)
    return V_exact

def H_0_exact(Energies):
    return np.diag(Energies)

def H_exact(h_x, V_exact, H_0_exact):
    # Calculate exact matrix
    return H_0_exact - h_x*V_exact

###############################################################################

def app_eigensystem(GS_indices, GS_energy, h_x_range, J, N, basis, Jij):
    # Calculate approximated eigenvalues and eigenstates for range(h_x)
    app_eigenvalues = np.zeros((len(GS_indices), len(h_x_range)))
    app_eigenstates = np.zeros((len(h_x_range), len(GS_indices), len(GS_indices)))
    
    H_0 = H_app_0(GS_energy, GS_indices)
    V = H_app_1(basis, GS_indices, N)
    H_2 = H_app_2(basis, Jij, GS_indices, N, GS_energy)
    
    for j, h_x in enumerate(h_x_range):
        app_eigenvalue, app_eigenstate = np.linalg.eigh(H_app(h_x, H_0, V, H_2, J));
        for i in range(len(GS_indices)):
            app_eigenvalues[i][j] = app_eigenvalue[i]
            for k in range(len(GS_indices)):
                app_eigenstates[j][i][k] = app_eigenstate[i][k]
    return app_eigenvalues, app_eigenstates

def exc_eigensystem(basis, h_x_range, lattice, Energies):
    # Calculate exact eigenvalues and eigenstates for range(h_x)
    exc_eigenvalues = np.zeros((basis.M, len(h_x_range)))
    exc_eigenstates = np.zeros((len(h_x_range), basis.M, basis.M))
    V_exc = V_exact(basis, lattice)
    H_0_exc = H_0_exact(Energies)
    for j, h_x in enumerate(h_x_range):
        exc_eigenvalue, exc_eigenstate = np.linalg.eigh(H_exact(h_x, V_exc, H_0_exc));
        for i in range(basis.M):
            exc_eigenvalues[i][j] = exc_eigenvalue[i]
            for k in range(basis.M):
                exc_eigenstates[j][i][k] = exc_eigenstate[i][k]
    return exc_eigenvalues, exc_eigenstates

###############################################################################
# For error analysis and curve fit
def poly_4(x, b, c):
    return b*x**3 + c*x**4;

def poly_3(x, b):
    # third order polynomial
    return b*x**3;

def poly_2(x, a):
    # second order polynomial
    return a*x**2;

def prob(eigenstate):
    norm = np.vdot(eigenstate, eigenstate)
    normed_eigenstate = eigenstate/(norm**0.5)
    return np.conjugate(normed_eigenstate)*normed_eigenstate

def prob_app(GS_indices, h_x_range, app_eigenstates):
    # Calculate probabilities for approximated eigenstates
    prob_app = np.zeros((len(GS_indices), len(h_x_range),))
    for j, h_x in enumerate(h_x_range):
        GS_prob_vector = prob(app_eigenstates[j][:, 0])
        for i in range(len(GS_indices)):
            prob_app[i][j] = GS_prob_vector[i]
    return prob_app

def prob_exc(GS_indices, h_x_range, exc_eigenstates):
    # Calculate probabilities for exact eigenstates
    prob_exc = np.zeros((len(GS_indices), len(h_x_range),))
    for j, h_x in enumerate(h_x_range):
        GS_prob_vector = prob(exc_eigenstates[j][:, 0])
        for k, i in enumerate(GS_indices):
            prob_exc[k][j] = GS_prob_vector[i]
    return prob_exc

def prob_exc_total(GS_indices, h_x_range, exc_eigenstates):
    # Probability of finding the system to be in each of the ground states
    prob_exc_total = np.zeros((len(GS_indices), len(GS_indices), len(h_x_range),))
    for n in range(len(GS_indices)):
        for j, h_x in enumerate(h_x_range):
            GS_prob_vector = prob(np.transpose(exc_eigenstates[j])[n])
            for k, i in enumerate(GS_indices):
                prob_exc_total[n][k][j] = GS_prob_vector[i]
    return prob_exc_total

def prob_excited_sum(GS_indices, h_x_range, prob_exc_total):
    # Probability of finding the system to be in excited states
    prob_excited_sum = np.zeros((len(GS_indices), len(h_x_range)))
    for n in range(len(GS_indices)):
        for i, h_x in enumerate(h_x_range):
            prob_excited_sum[n][i] = 1 - np.sum(prob_exc_total[n][:, i])
    return prob_excited_sum

def normalize(eigenstate):
    # Normalize
    norm = np.vdot(eigenstate, eigenstate)**0.5
    return eigenstate/norm

def GS_exc_eigenstates(GS_indices, h_x_range, exc_eigenstates):
    GS_exc_eigenstates = np.zeros((len(h_x_range), len(GS_indices), len(GS_indices)))
    for j in range(len(h_x_range)):
        for n, m in enumerate(GS_indices):
            for i, k in enumerate(GS_indices):
                GS_exc_eigenstates[j, n, i] = exc_eigenstates[j, m, i]
    return GS_exc_eigenstates

def norm_GS_exc_eigenstates(GS_indices, h_x_range, exc_eigenstates):
    # Renormalize
    normalized_GS_exc_eigenstates = np.zeros((len(h_x_range), len(GS_indices), len(GS_indices)))
    GS_exc_ES = GS_exc_eigenstates(GS_indices, h_x_range, exc_eigenstates)
    for j in range(len(h_x_range)):
        for n in range(len(GS_indices)):
            normed_vector = normalize(GS_exc_ES[j, :, n])
            for i in range(len(GS_indices)):
                normalized_GS_exc_eigenstates[j, i, n] = normed_vector[i]
    return normalized_GS_exc_eigenstates

def fidelity(exc_eigenstate, app_eigenstate):
    # Calculate fidelity
    dot = np.vdot(exc_eigenstate, np.conjugate(app_eigenstate))
    return dot*np.conjugate(dot)

def sort(lst):
    # identify degenerate energy level and resort
    epsilon = 1*10**(-12)
    order = [];
    floor = np.array([0]);
    for i in range(1, len(lst)):
        if abs(lst[i-1] - lst[i]) <= epsilon:
            floor = np.append(floor, i)
        else:  
            order.append(floor)
            floor = np.array([i]);
    order.append(floor)
    return list(order) 

def fidelity_array(GS_indices, h_x_range, GS_exc_eigenvalues, app_eigenvalues, exc_eigenstates, app_eigenstates):
    # Produce an array of fidelities between exc and app to be plotted
    fidelity_array = np.zeros((len(GS_indices), len(h_x_range)))
    GS_exc_ES = GS_exc_eigenstates(GS_indices, h_x_range, exc_eigenstates)
    for i in range(len(h_x_range)):
        sorted_exc_energy_indices = sort(GS_exc_eigenvalues[:, i])
        sorted_app_energy_indices = sort(app_eigenvalues[:, i])
        for l, app_level in enumerate(sorted_app_energy_indices):
            for j in app_level:
                fidel_sum = 0;
                for k in sorted_exc_energy_indices[l]:
                    fidel = fidelity(GS_exc_ES[i, :, k], app_eigenstates[i, :, j])
                    fidel_sum += fidel
                fidelity_array[j, i] = fidel_sum
    return fidelity_array

def infidelity_array(fidelity_array):
    return 1 - fidelity_array