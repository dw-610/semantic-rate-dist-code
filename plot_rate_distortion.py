"""
This script sweeps over a range of beta values for the Blahut-Ariomoto alg to
return the corresponding points on the R(D) curve.
"""

# ------------------------------------------------------------------------------
# imports

import numpy as np
import matplotlib.pyplot as plt

from modules import infotheory as it

# ------------------------------------------------------------------------------

def main():

    N: int = 4 # size of the input alphabet
    SEED: int = None # random seed for reproducability
    TOL: float = 1e-3 # algorithm tolerance
    MAX_ITER: int = 1000 # maximum alg iterations before breaking

    if SEED:
        np.random.seed(SEED)

    p_x = np.random.random(size=(1,N))
    p_x /= np.sum(p_x)

    d_x_y = np.random.random((N,N)) # d_x_hat[i,j] = d(x_i, \hat{x}_j)
    
    betas = np.logspace(np.log10(1), np.log10(100), 100)
    E_Ds, R_Ds = [], []
    for beta in betas:
        p_y_given_x = it.blahut_arimoto(p_x, d_x_y, beta, TOL, MAX_ITER)
        p_x_y = it.get_joint_dist(p_x, p_y_given_x)
        E_Ds.append(it.get_expected_distortion(p_x_y, d_x_y))
        R_Ds.append(it.get_mutual_info(p_x_y))
    
    plt.figure()
    plt.plot(E_Ds, R_Ds)
    plt.xlabel('Distortion (D)')
    plt.ylabel('R(D)')
    plt.grid()

    plt.figure()
    plt.plot(betas, E_Ds)
    plt.xlabel('beta')
    plt.ylabel('Distortion (D)')
    plt.grid()
    plt.show()

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    main()

# ------------------------------------------------------------------------------