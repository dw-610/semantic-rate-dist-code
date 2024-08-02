"""
This script does some experiments with the "typical" coding scheme from the
rate distortion achievability proof.

Runs Monte Carlo trials to approximate the probability of a random sequence
having a "typical codeword" pair for a randomly generated codebook.
"""

# ------------------------------------------------------------------------------
# imports

import numpy as np
import multiprocessing as mp

from modules import infotheory

# ------------------------------------------------------------------------------

def typical_test(x_seq, y_seq, p_x_dict, p_y_dict, p_x_y_dict, d_x_y, H_x,
                 H_y, H_x_y, E_d, eps):
    """Tests to see if two sequences are distortion-typical."""
    n = len(x_seq)
    H_x_emp = 0.0
    H_y_emp = 0.0
    H_x_y_emp = 0.0
    d_x_y_emp = 0.0
    for i in range(n):
        x = x_seq[i]
        y = y_seq[i]
        if p_x_dict[x] != 0:
            H_x_emp += -1/n*np.log2(p_x_dict[x_seq[i]])
        if p_y_dict[y] != 0:
            H_y_emp += -1/n*np.log2(p_y_dict[y_seq[i]])
        if p_x_y_dict[(x, y)] != 0:
            H_x_y_emp += -(1/n)*np.log2(p_x_y_dict[(x, y)])
        d_x_y_emp += (1/n)*d_x_y[x, y]

    if np.abs(H_x - H_x_emp) < eps and np.abs(H_y - H_y_emp) < eps \
        and np.abs(H_x_y - H_x_y_emp) < eps and np.abs(E_d - d_x_y_emp) < eps:
        return True
    else:
        return False
    
def test_codeword(seq, N, n, p_hat, chunk_size, *args):
    for _ in range(chunk_size):
        codeword = np.random.choice(N, (n,), p=p_hat)
        if typical_test(seq, codeword, *args):
            return True
    return False
    
def test_codebook(seq, N, n, R, p_hat, *args):
    """Tests to see if a codebook contains a typical sequence for a sequence."""
    num_codewords = 2**(n*R)
    num_workers = mp.cpu_count()
    chunk_size = num_codewords // num_workers
    with mp.Pool(num_workers) as pool:
        results = pool.starmap(
            test_codeword,
            [(seq, N, n, p_hat, chunk_size, *args) for _ in range(num_workers)]
        )
    return any(results)

def mc_test_codebooks(p_x, p_hat, N, n, R, n_sims, *args):
    """Monte-carlo simulations to estimate prob of codebook with typ. cw."""
    hits = 0
    for i in range(n_sims):
        rng = np.random.default_rng()
        seq = rng.choice(N, size=(n,), p=p_hat)
        # seq = np.random.choice(N, (n,), p=p_x)
        hits += int(test_codebook(seq, N, n, R, p_hat, *args))
        if (i+1) % 1 == 0:
            print(f'({i+1}/{n_sims}) Hits: {hits}')
    return hits/n_sims

# ------------------------------------------------------------------------------

def main():

    SEED = None # seed for random number generator

    N = 4 # source alphabet size
    R = 1 # rate of the code
    n = 10 # sequence length

    # parameters for the Blahu-Arimoto alg
    BETA = 1
    TOL = 1e-4
    MAX_ITER = 1000

    EPS: float = 1.0 # typicality threshold

    N_SIMS: int = 100 # number of Monte-Carlo trials

    if SEED:
        np.random.seed(SEED)
    
    # generate the source alphabet and distribution
    X = np.array(range(N)).reshape((1,N))
    p_x = np.random.random(X.shape)
    p_x /= np.sum(p_x)

    # generator random distortion values
    d_x_hat = np.random.random((N,N)) # d_x_hat[i,j] = d(x_i, \hat{x}_j)

    # get the optimal conditional dist -> joint and other marginal dists
    p_hat_given_x = infotheory.blahut_arimoto(p_x, d_x_hat, BETA, TOL, MAX_ITER)
    p_x_hat = infotheory.get_joint_dist(p_x, p_hat_given_x)
    _, p_hat = infotheory.get_marginal_dists(p_x_hat)

    # compute entropy for all the distributions
    H_x, H_hat = infotheory.get_entropy(p_x), infotheory.get_entropy(p_hat)
    H_x_hat = infotheory.get_joint_entropy(p_x_hat)

    # compute the expected distortion for the joint dist
    E_d = infotheory.get_expected_distortion(p_x_hat, d_x_hat)

    # generate dictionaries for quickly mapping between alphabets/dists
    p_x_dict = {x: p_x for x, p_x in zip(X.flatten(), p_x.flatten())}
    p_hat_dict = {hat: p_hat for hat, p_hat in zip(X.flatten(), p_hat.flatten())}
    p_x_hat_dict = {}
    for i in range(N):
        for j in range(N):
            p_x_hat_dict[(X.flatten()[i], X.flatten()[j])] = p_x_hat[i,j]

    # run Monte Carlo trials and print results
    p_typical_cw = mc_test_codebooks(
        p_x.flatten(), p_hat.flatten(), N, n, R, N_SIMS, p_x_dict, p_hat_dict, 
        p_x_hat_dict, d_x_hat, H_x, H_hat, H_x_hat, E_d, EPS)
    print('Approximate probability of a random codebook having a typical')
    print(f'codeword for a random source sequence: {p_typical_cw}')

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    main()

# ------------------------------------------------------------------------------