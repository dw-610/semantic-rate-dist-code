"""
This script simulates a goal-oriented semantic communication system, i.e., the
one modeled by the semantic-funcational rate distortion problem.

Not using an optimal technical code for now, just going to use *some* code.

Assumes that the task codebook is the same as the task alphabet.
"""

# ------------------------------------------------------------------------------
# imports

from itertools import product

import numpy as np
from numpy.random import normal as randn
from numpy.random import random_sample as rand

from modules.sims import lloyd_technical_random_decoder as lloyd_random
from modules.sims import lloyd_technical_optimal_decoder as lloyd_optim

import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------

def main():
    
    N = 10 # alphabet size
    M = 4 # conceptual space dimensionality
    K = 2 # task alphabet size
    # n = 4 # sequence length

    R = 2 # rate of the code

    VERBOSE = False

    INNER_SIMS = 100
    OUTER_SIMS = 100

    ns = [1, 2, 3]

    X = [x for x in 'abcdefghij']
    U = [u for u in ['pass', 'fail']]

    avg_d_s_list, avg_d_f_list, avg_Delta_list = [], [], []
    for n in ns:
        d_s_list, d_f_list, Delta_list = [], [], []
        for i in range(OUTER_SIMS):
            p_x = rand(N)
            p_x /= np.sum(p_x)

            func_dist = {(x, u): rand() for x, u in product(X, U)}

            # d_s, d_f, Delta = lloyd_random(X, p_x, n, M, U, func_dist, R, 
            #                                INNER_SIMS, VERBOSE, min_iters=100)

            # print(f'Avg. Semantic Distortion:    {d_s:.3f}')
            # print(f'Avg. Functional Distortion:  {d_f:.3f}')
            # print(f'Avg. Distortion Discrepancy: {Delta:.3f}')

            d_s, d_f, Delta = lloyd_optim(X, p_x, n, M, U, func_dist, R, INNER_SIMS, 
                                        VERBOSE, min_iters=100)
            d_s_list.append(d_s)
            d_f_list.append(d_f)
            Delta_list.append(Delta)

            # print(f'\nAvg. Semantic Distortion:    {d_s:.3f}')
            # print(f'Avg. Functional Distortion:  {d_f:.3f}')
            # print(f'Avg. Distortion Discrepancy: {Delta:.3f}')

            print(f'\r{i+1}/{OUTER_SIMS}', end='')
        print(f'\nDone with n={n}\n')
        avg_d_s_list.append(sum(d_s_list)/OUTER_SIMS)
        avg_d_f_list.append(sum(d_f_list)/OUTER_SIMS)
        avg_Delta_list.append(sum(Delta_list)/OUTER_SIMS)

    plt.plot(ns, avg_d_s_list, label='Avg. delta')
    plt.plot(ns, avg_d_f_list, label='Avg. d_f')
    plt.plot(ns, avg_Delta_list, label='Avg. Delta')
    plt.grid()
    plt.legend()
    plt.show()

    

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    main()

# ------------------------------------------------------------------------------