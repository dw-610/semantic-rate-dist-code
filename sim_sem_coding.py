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

from modules.sims import naive_technical_random_decoder as naive_random
from modules.sims import lloyd_technical_random_decoder as lloyd_random
from modules.sims import lloyd_technical_optimal_decoder as lloyd_optim

# ------------------------------------------------------------------------------

def main():
    
    N = 100 # alphabet size
    M = 5 # conceptual space dimensionality
    K = 20 # task alphabet size

    R = 4 # rate of the code

    VERBOSE = False

    SIMS = 1000

    SEED = 42 # seed for random number generator
    # if SEED is not None:
    #     np.random.seed(SEED)

    X = [x for x in range(N)]
    U = [u for u in range(K)]
    Z = {i: randn(size=(M,)) for i in range(N)}
    
    p_x = rand(N)
    p_x /= np.sum(p_x)
    func_dist = {(x, u): 4*rand() for x, u in product(X, U)}

    if SEED is not None:
        np.random.seed(SEED)
    d_s, d_f, Delta = naive_random(X, p_x, Z, U, func_dist, R, SIMS, VERBOSE)

    print(f'Avg. Semantic Distortion:    {d_s:.2f}')
    print(f'Avg. Functional Distortion:  {d_f:.2f}')
    print(f'Avg. Distortion Discrepancy: {Delta:.2f}')

    if SEED is not None:
        np.random.seed(SEED)
    d_s, d_f, Delta = lloyd_random(X, p_x, Z, U, func_dist, R, SIMS, VERBOSE, 
                                   min_iters=100)

    print(f'\nAvg. Semantic Distortion:    {d_s:.2f}')
    print(f'Avg. Functional Distortion:  {d_f:.2f}')
    print(f'Avg. Distortion Discrepancy: {Delta:.2f}')

    if SEED is not None:
        np.random.seed(SEED)
    d_s, d_f, Delta = lloyd_optim(X, p_x, Z, U, func_dist, R, SIMS, VERBOSE, 
                                  min_iters=100)

    print(f'\nAvg. Semantic Distortion:    {d_s:.2f}')
    print(f'Avg. Functional Distortion:  {d_f:.2f}')
    print(f'Avg. Distortion Discrepancy: {Delta:.2f}')

    

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    main()

# ------------------------------------------------------------------------------