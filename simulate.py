"""
This script simulates a goal-oriented semantic communication system, i.e., the
one modeled by the semantic-funcational rate distortion problem.

Assumes that the task codebook is the same as the task alphabet.
"""

# ------------------------------------------------------------------------------
# imports

from itertools import product
import numpy as np
from numpy.random import random_sample as rand

from modules.sims import simulate_system as sim

# ------------------------------------------------------------------------------

def main():
    
    # PARAMETERS
    N = 12          # alphabet size
    PX = 'uniform'  # type of source distribution ('uniform' or 'random')
    M = 2           # conceptual space dimensionality
    K = 4           # task alphabet size
    n = 1           # block sequence length
    R = 3           # rate of the code

    SIMS = 100000    # number of Monte Carlo simulations to run

    PRINT_LATEX = False

    # random seeds for reproducibility
    PX_SEED = 610           # seed for choosing p(x) (if PX = 'random')
    DF_SEED = 67            # seed for choosing distortion value
    SEM_ENC_SEED = 27       # seed for choosing semantic representations
    LLOYD_SEED = 87         # seed for running Lloyd's algorithm
    SEM_DEC_SEED = 1014     # seed for choosing the (random) semantic decoder

    # set true to print additional output (only works for n = 1)
    VERBOSE = False

    X = [x for x in range(N)]
    U = [u for u in range(K)]

    if PX == 'uniform':
        p_x = np.ones(N)/N
    elif PX == 'random':
        random_state = np.random.get_state()
        np.random.seed(PX_SEED)
        p_x = np.random.random(N)
        p_x /= np.sum(p_x)
        np.random.set_state(random_state)
    else:
        raise ValueError('Invalid type of source distribution PX')
    
    if DF_SEED is not None:
        random_state = np.random.get_state()
        np.random.seed(DF_SEED)
        func_dist = {(x, u): rand() for x, u in product(X, U)}
        np.random.set_state(random_state)
    else:
        func_dist = {(x, u): rand() for x, u in product(X, U)}

    # carry out simulations with the optimal semantic decoder
    d_s, d_f, Delta = sim(X, p_x, n, M, U, func_dist, R, 'min_discrepancy',
                            SIMS, VERBOSE, SEM_ENC_SEED, LLOYD_SEED, 
                            SEM_DEC_SEED)
    
    # carry out simulations with the oracle semantic decoder
    d_s_o, d_f_o, Delta_o = sim(X, p_x, n, M, U, func_dist, R, 'oracle',
                            SIMS, VERBOSE, SEM_ENC_SEED, LLOYD_SEED, 
                            SEM_DEC_SEED)
    
    # carry out simulations with the random semantic decoder
    d_s_r, d_f_r, Delta_r = sim(X, p_x, n, M, U, func_dist, R, 'random',
                            SIMS, VERBOSE, SEM_ENC_SEED, LLOYD_SEED,
                            SEM_DEC_SEED)

    print('\nN', 'p', 'M', 'K', 'n', 'R')
    print(N, PX, M, K, n, R)
    print('\n           Edel ', 'EDel ', 'Ed_f')
    print(f'(Proposed) {d_s:.3f}', f'{Delta:.3f}', f'{d_f:.3f}')
    print(f'(Naive)    {d_s_r:.3f}', f'{Delta_r:.3f}', f'{d_f_r:.3f}')
    print(f'(Oracle)   {d_s_o:.3f}', f'{Delta_o:.3f}', f'{d_f_o:.3f}')

    # print out parameters and results in LaTeX table row format
    if PRINT_LATEX:
        print('\nfor pasting in to latex table row:')
        print(N, PX, M, K, n, R, f'{d_s:.3f}', f'{Delta:.3f}', f'{d_f:.3f}',
                f'{Delta_r:.3f}',f'{d_f_r:.3f}',f'{Delta_o:.3f}',f'{d_f_o:.3f}',
                sep=' & ', end = ' \\\\ \n')

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    main()

# ------------------------------------------------------------------------------