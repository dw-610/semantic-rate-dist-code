"""
This script carries out typical coding at a specified rate for a given source
alphabet. Measures the expected distortion.

Assumes the input and output alphabet are the same.
"""

# ------------------------------------------------------------------------------
# imports

import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

# ------------------------------------------------------------------------------

def get_joint_dist(p_x: np.ndarray, p_y_given_x: np.ndarray) -> np.ndarray:
    """
    Computes the joint distribution p(x,y) given the marginal p(x) and the 
    conditional p(y|x).

    Parameters
    ----------
    p_x : np.ndarray
        The marginal distribution of x, specified as a 1xN vector.
        Entries should sum to 1.
    p_y_given_x : np.ndarray
        The conditional distribution of y given x, specified as a MxN matrix,
        where the [i,j]th entry denotes p(y_i | x_j). Columns should sum to 1.

    Returns
    -------
    np.ndarray
        The joint distribution, returns as a NxM matrix, where the [i,j]th entry
        denotes p(x_i, y_j).
    """
    return (p_y_given_x*p_x).T

def get_marginal_dists(p_x_y: np.ndarray) -> np.ndarray:
    """
    Computes the marginal distribution p(y) given the marginal p(x,y).

    Parameters
    ----------
    p_x_y : np.ndarray
        The joint distribution p(x,y), specified as a NxM matrix. The [i,j]th 
        entry corresponds to p(x_i, y_j).

    Returns
    -------
    np.ndarray, np.ndarray
        A tuple containing the marginal distribution (p(x), p(y)), which are 
        N-dim and M-dim 1D vectors, respectively.
    """
    p_x = np.sum(p_x_y, axis=1)
    p_y = np.sum(p_x_y, axis=0)
    return p_x, p_y

def get_mutual_info(p_x_y: np.ndarray) -> float:
    """
    Computes the mutual information I(X;Y) between X, Y ~ p(x,y).

    Parameters
    ----------
    p_x_y : np.ndarray
        The joint distribution p(x,y), specified as a NxM matrix. The [i,j]th 
        entry corresponds to p(x_i, y_j).

    Returns
    -------
    float
        The mutual information I(X;Y).
    """
    p_x, p_y = get_marginal_dists(p_x_y)
    I_X_Y = 0
    for i in range(len(p_x)):
        for j in range(len(p_y)):
            if p_x_y[i,j] == 0:
                pass
            else:
                I_X_Y += p_x_y[i,j] * np.log2(p_x_y[i,j]/p_x[i]/p_y[j])
    return float(I_X_Y)

def get_entropy(p_x: np.ndarray) -> float:
    """
    Computes the entropy H(X) of a source X ~ p(x).

    Parameters
    ----------
    p_x : np.ndarray
        The marginal distribution of X, specified as a 1D vector.

    Returns
    -------
    float
        The entropy H(X).
    """ 
    return float(-p_x @ np.log2(p_x))

def get_joint_entropy(p_x_y: np.ndarray) -> float:
    """
    Computes the joint entropy H(X,Y) of a source X, Y ~ p(x,y).

    Parameters
    ----------
    p_x_y : np.ndarray
        The joint distribution p(x,y), specified as a NxM matrix. The [i,j]th 
        entry corresponds to p(x_i, y_j).

    Returns
    -------
    float
        The entropy H(X,Y).
    """ 
    return float(-p_x_y.flatten() @ np.log2(p_x_y.flatten()))

def blahut_arimoto(p_x: np.ndarray, d_x_y: np.ndarray, beta: float = 1.0,
                   tol: float = 1e-3, max_iter: int = 100) -> np.ndarray:
    """
    Implements the Blahut-Arimoto algorithm for finding the optimal conditional
    distribution that minimizes mutual info given a distortion constraint.
    Further information on 
    [Wikipedia](https://en.wikipedia.org/wiki/Blahut-Arimoto_algorithm).

    This implementation assumes the size of the source and codebook alphabets
    are the same (i.e., conditional dist matrix is square).

    Parameters
    """
    N = len(p_x)

    # initialize a random conditional distribution
    p_y_given_x = np.random.random((N,N))
    p_y_given_x /= np.sum(p_y_given_x, axis=0, keepdims=True)

    iter = 0
    delta = np.Inf
    while delta > tol and iter < max_iter:
        iter += 1
        old_p_y_given_x = np.copy(p_y_given_x)
        
        # update the marginal of Y
        p_y = p_y_given_x @ p_x

        # update the conditional distribution of Y|X
        for i in range(N):
            for j in range(N):
                num = p_y[j]*np.exp(-beta*d_x_y[i,j])
                denom = 0
                for k in range(N):
                    denom += p_y[k]*np.exp(-beta*d_x_y[i,k])
                p_y_given_x[j,i] = num/denom

        # quantify the change
        delta = np.linalg.norm(old_p_y_given_x - p_y_given_x)
        # print(f'Iteration {iter}: delta = {delta:.5f}')

    if iter == max_iter:
        print(f'Did not converge in {max_iter} iterations.')
    else:
        print(f'Converged in {iter} iterations.')

    return p_y_given_x

def sweep_blahut_arimoto(p_x: np.ndarray, d_x_y: np.ndarray, D: float,
                   tol: float = 1e-3, max_iter: int = 100) -> np.ndarray:
    """
    Performs a sweep over the Blahut-Arimoto-returned p(y|x) distributions,
    finding the one that just meets the expected distortion threshold D.
    """

    # start with a low beta and increase it until E_D < D
    beta = 1e-5
    E_D = 1e9
    iter = 0
    while E_D > D and iter < 1e9:
        iter += 1
        beta *= 1.1
        p_y_given_x = blahut_arimoto(p_x, d_x_y, beta, tol, max_iter)
        p_x_y = get_joint_dist(p_x, p_y_given_x)
        E_D = get_expected_distortion(p_x_y, d_x_y)
        print(f'Iter: {iter}: beta = {beta}, E(D) = {E_D}')
    return p_y_given_x

def plot_rate_distortion_function(p_x: np.ndarray, d_x_y: np.ndarray, D: float,
                   tol: float = 1e-3, max_iter: int = 100) -> np.ndarray:
    """
    Performs a sweep over the Blahut-Arimoto-returned p(y|x) distributions,
    computing the expected distortion and minimum rate for each to return R(D).
    """
    betas = np.logspace(np.log10(1), np.log10(100), 100)
    E_Ds, R_Ds = [], []
    for beta in betas:
        p_y_given_x = blahut_arimoto(p_x, d_x_y, beta, tol, max_iter)
        p_x_y = get_joint_dist(p_x, p_y_given_x)
        E_Ds.append(get_expected_distortion(p_x_y, d_x_y))
        R_Ds.append(get_mutual_info(p_x_y))
    
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

def get_expected_distortion(p_x_y: np.ndarray, d_x_y: np.ndarray):
    """
    Computes the expected distortion givena a joint distribution and distortion
    function.
    """
    return np.sum(p_x_y*d_x_y)

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

def main(alphabet_size: int, sequence_len: int = 1, seed: int = None):

    N = alphabet_size
    BETA = 40
    D = 0.15 # distortion threshold
    R = 1 
    n = 10

    EPS: float = 1.0

    N_SIMS: int = 100

    if seed:
        np.random.seed(seed)
    
    X = np.array(range(N))
    p_x = np.random.random(X.shape)
    p_x /= np.sum(p_x)

    d_x_hat = np.random.random((N,N)) # d_x_hat[i,j] = d(x_i, \hat{x}_j)

    # plot_rate_distortion_function(p_x, d_x_hat, D, 1e-4, 1000)

    p_hat_given_x = blahut_arimoto(p_x, d_x_hat, BETA, 1e-4, 1000)
    p_x_hat = get_joint_dist(p_x, p_hat_given_x)
    _, p_hat = get_marginal_dists(p_x_hat)

    H_x, H_hat = get_entropy(p_x), get_entropy(p_hat)
    H_x_hat = get_joint_entropy(p_x_hat)

    E_d = get_expected_distortion(p_x_hat, d_x_hat)

    p_x_dict = {x: p_x for x, p_x in zip(X, p_x)}
    p_hat_dict = {hat: p_hat for hat, p_hat in zip(X, p_hat)}
    p_x_hat_dict = {}
    for i in range(N):
        for j in range(N):
            p_x_hat_dict[(X[i], X[j])] = p_x_hat[i,j]

    # codebook = {i: np.random.choice(N, (n,), p=p_hat) for i in range(2**(n*R))}

    # seq = np.random.choice(N, (n,), p=p_x)

    # if test_codebook(seq, codebook, p_x_dict, p_hat_dict, p_x_hat_dict, d_x_hat, 
    #              H_x, H_hat, H_x_hat, E_d, EPS):
    #     print('Found a typical codeword!')
    # else:
    #     print('No typical codeword in the codebook.')

    p_typical_cw = mc_test_codebooks(
        p_x, p_hat, N, n, R, N_SIMS, p_x_dict, p_hat_dict, p_x_hat_dict, d_x_hat, H_x, 
        H_hat, H_x_hat, E_d, EPS)
    print('Approximate probability of a random codebook having a typical')
    print(f'codeword for a random source sequence: {p_typical_cw}')

    



    

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    
    N: int = 4 # size of the input alphabet
    SEED: int = None # random seed for reproducability

    main(N, seed=SEED)

# ------------------------------------------------------------------------------