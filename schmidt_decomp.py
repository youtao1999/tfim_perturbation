import tfim_rdm
import numpy as np
import matplotlib.pyplot as pl

def partition_basis(basis, GS_indices, A, B):
    
    # building initial basis as set since to avoid repeated elements
    A_basis = set()
    B_basis = set()
    
    for index, GS_index in np.ndenumerate(GS_indices):
        state = basis.state(GS_index)
        A_basis.add(tuple(state[A]))
        B_basis.add(tuple(state[B]))
    
    def sum_digits(digits): 
        return sum(c << i for i, c in enumerate(digits)) 
    
    # now we extract the elements from this set and start building the ordered
    # basis
    
    # reordering basis A
    index_matching_A = {}

    for ele in A_basis:
        ele = np.array(ele)
        index_matching_A[sum_digits(ele)] = np.array(ele)

    A_reordered_basis = np.zeros((len(index_matching_A),len(list(index_matching_A.values())[0])))
    for index, key in enumerate(index_matching_A.keys()):
        A_reordered_basis[index] = index_matching_A[key]
    
    # reordering basis B
    index_matching_B = {}

    for ele in B_basis:
        ele = np.array(ele)
        index_matching_B[sum_digits(ele)] = np.array(ele)

    B_reordered_basis = np.zeros((len(index_matching_B),len(list(index_matching_B.values())[0])))
    for index, key in enumerate(index_matching_B.keys()):
        B_reordered_basis[index] = index_matching_B[key]
        
    return A_reordered_basis, B_reordered_basis

def beta_ij(basis, GS_indices, A, B, overall_GS, perturbation_param_index):
    # this function builds beta_ij matrix as a function of the index of the perturbation parameter

    # build A, B basis first
    A_basis, B_basis = partition_basis(basis, GS_indices, A, B)
    s = overall_GS[perturbation_param_index]

    def find(basis, target):
        # this function finds the index of the target state in the basis state, gives the index of 1 in BETA
        for i, state in enumerate(basis):
            if np.array_equiv(state, target):
                return i

    BETA = np.zeros((len(A_basis),len(B_basis)))

    for (probability, GS_index) in zip(s, GS_indices):
        GS_state = basis.state(GS_index)
        i = find(A_basis, GS_state[A])
        j = find(B_basis, GS_state[B])
        BETA[i, j] += probability
    return BETA

def perturb_entropy(basis, GS_indices, A, B, overall_GS, perturbation_param_index):
    
    BETA = beta_ij(basis, GS_indices, A, B, overall_GS, perturbation_param_index)

    # perform uv decomposition

    u, s, vh = np.linalg.svd(BETA, full_matrices=True)
    
    s = s[np.where(s != 0.)]
    # add conditional statement to remove all the zero singular values

    entropy = -np.dot(s**2, np.log(s**2))
    
    return entropy

def exc_entropy(basis, exc_eigenstates, exc_eigenvalues, A, B, perturbation_param_index):
    psi0 = exc_eigenstates[ :, :, np.argmin(exc_eigenvalues.T[5])]
    S, U, V = tfim_rdm.svd(basis, A, B, psi0[perturbation_param_index])
    entropy = tfim_rdm.entropy(S)
    return entropy

def entropy_analysis(basis, exc_eigenstates, exc_eigenvalues, GS_indices, A, B, overall_GS, h_x_range):
    
    # This function prints out the entropy analysis plots comparing perturbation 
    # theory prediction and exact diagonalization methods

    perturb_entropy_arr = np.zeros(len(h_x_range))
    exc_entropy_arr = np.zeros(len(h_x_range))

    for i in range(len(h_x_range)):
        perturb_entropy_arr[i] = schmidt_decomp.perturb_entropy(basis, GS_indices, A, B, overall_GS, i)
        exc_entropy_arr[i] = schmidt_decomp.exc_entropy(basis, exc_eigenstates, exc_eigenvalues, A, B, i)

    # convergence plot
    fig = pl.figure(figsize=(8, 6))
    pl.rcParams['font.size'] = '18'
    pl.plot(h_x_range,exc_entropy_arr, lw=1.3, ls='-', color="blue", label= "exact diagonalization")
    pl.plot(h_x_range,perturb_entropy_arr, lw=1.3, ls='-', color="green", label = "perturbation theory prediction")
    pl.ylabel('entropy', fontsize=18)
    pl.xlabel('perturbation parameter', fontsize=18)
    # pl.axis([np.power(10, sigma_v)[lowerbound], np.power(10, sigma_v)[upperbound], chisq[lowerbound], chisq[upperbound]])
    pl.xticks(fontsize=18)
    pl.yticks(fontsize=18)
    pl.tick_params('both', length=7, width=2, which='major')
    pl.tick_params('both', length=5, width=2, which='minor')
    pl.grid(False)
    pl.xscale('log')
    pl.legend(loc=5, prop={'size': 16}, numpoints=1, scatterpoints=1, ncol=1)
    fig.tight_layout(pad=0.5)

    # error plot

    # Boundary index for the plot
    fig = pl.figure(figsize=(8, 6))
    pl.rcParams['font.size'] = '18'
    pl.plot(h_x_range,abs(exc_entropy_arr-perturb_entropy_arr), lw=1.3, ls='-', color="blue", label= "entropy error")
    pl.ylabel('entropy error', fontsize=18)
    pl.xlabel('perturbation parameter', fontsize=18)
    # pl.axis([np.power(10, sigma_v)[lowerbound], np.power(10, sigma_v)[upperbound], chisq[lowerbound], chisq[upperbound]])
    pl.xticks(fontsize=18)
    pl.yticks(fontsize=18)
    pl.tick_params('both', length=7, width=2, which='major')
    pl.tick_params('both', length=5, width=2, which='minor')
    pl.grid(False)
    pl.xscale('log')
    pl.yscale('log')
    pl.legend(loc=4, prop={'size': 16}, numpoints=1, scatterpoints=1, ncol=1)
    fig.tight_layout(pad=0.5)