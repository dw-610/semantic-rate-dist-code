"""
This module contains high-level simulation routines.
"""

# ------------------------------------------------------------------------------

import numpy as np
from numpy.random import choice as choose

from . import coding as cd
from .coding import semantic_distortion as sem_dist
from .coding import extend_list_to_seq as extend_list
from .utils import print_code

# ------------------------------------------------------------------------------

def sim_comm(x: list, code: dict) -> tuple:
    """
    Simulates a single communication sequence, from the generation of a source
    symbol x to the output of the estimated task variable u_hat.
    
    Parameters
    ----------
    x : list[Any] (elements of the source alphabet X)
        Sequence of n source symbols that is the input to the system.
    codes : dict
        A dictionary containing keyword pairs for the code functions. Should
        have the following keys/value pairs:
        - 'e_s': dict (semantic encoder)
        - 'e_t': dict (technical encoder)
        - 'g_t': dict (technical decoder)
        - 'g_s': dict (semantic decoder)
    
    Returns
    -------
    uh : Any
        The output sequence of the communication chain for the input x.
    z_id : int
        IDs of the semantic representation sequence for x.
    zh_id : int
        IDs of the recovered semantic representation sequence.
    """
    e_s, e_t, g_t, g_s = [code[k] for k in ('e_s', 'e_t', 'g_t', 'g_s')]
    
    z_id    = e_s[x]
    cw      = e_t[z_id]
    zh_id   = g_t[cw]
    uh      = g_s[zh_id]

    return uh, z_id, zh_id

# ------------------------------------------------------------------------------

def naive_technical_random_decoder(
        X: list, p_x: np.ndarray, n: int, M: int, U: list, func_dist: dict, 
        rate: int, num_sims: int, verbose = False):
    """
    DEPRECATED - NOT GOING TO USE THIS IN SIMULATIONS

    Simulates a naive technical code that randomly assigns codewords as members
    of the source alphabet, and a semantic decoder that randomly maps semantic
    codewords to members of the task alphabet.

    Parameters
    ----------
    X : list
        X : list[any]
        List of inputs alphabet symbols.
    p_z : np.ndarray
        The probability distribution of the source.
        NumPy array with shape (len(X),) where the elements sum to 1.
    n : int
        The block length of transmitted sequences.
    M : int
        The dimensionality of the semantic space.
    U : list
        List of task alphabet symbols.
    func_dist : dict
        Function mapping (X,U) pairs to distortion values, in the form of a 
        dictionary with keys (x,u), x in X and u in U, and values as floats.
        Every possible pair must be included in the dictionary.
    rate : int
        The rate of the technical code in bits/semantic representation.
    num_sums : int
        The number of Monte Carlo simulations to run.
    
    Returns
    -------
    avg_sem_dist : float
        The average semantic distortion over the simulations.
    avg_fun_dist : float
        The average functional distortion over the simulations.
    avg_discrp : float
        The average distortion discrepancy over the simulations.
    """
    print('\n**WARNING** naive_technical_random_decoder is deprecated.',
          'Please use another simulation routine.\n')
    return None
    N = len(X)

    g_s, Z = cd.get_semantic_encoder(X, M, n)
    Zh_set = {i: Z[i] for i in range(N)}
    code = {
        'e_s': cd.get_semantic_encoder(X, Z),
        **dict(zip(['e_t', 'g_t'], 
                   cd.get_technical_code(Z, Zh_set, rate))),
        'g_s': cd.get_semantic_decoder(Zh_set, U)
    }
    if verbose: print_code(code, Z, Zh_set)

    if verbose: print('\nBeginning naive-random simulations...')
    cum_sem_dist, cum_fun_dist, cum_discrep = 0, 0, 0
    for i in range(num_sims):
        x = X[choose(N, p=p_x)]
        uh, z_id, zh_id = sim_comm(x, code)
        try:
            d_s = sem_dist(Z[z_id], Zh_set[zh_id])
        except KeyError:
            breakpoint()
        d_f = func_dist[(x, uh)]
        cum_sem_dist += d_s
        cum_fun_dist += d_f
        cum_discrep += (d_s - d_f)**2
    if verbose: print('Done!')

    avg_sem_dist = cum_sem_dist/num_sims
    avg_fun_dist = cum_fun_dist/num_sims
    avg_discrp = cum_discrep/num_sims

    return avg_sem_dist, avg_fun_dist, avg_discrp

# ------------------------------------------------------------------------------

def lloyd_technical_random_decoder(
        X: list, p_x: np.ndarray, n: int, M: int, U: list, func_dist: dict, 
        rate: int, num_sims: int, verbose = False, **kwargs):
    """
    Simulates a naive technical code that randomly assigns codewords as members
    of the source alphabet, and a semantic decoder that randomly maps semantic
    codewords to members of the task alphabet.

    Parameters
    ----------
    X : list
        X : list[any]
        List of inputs alphabet symbols.
    p_z : np.ndarray
        The probability distribution of the source.
        NumPy array with shape (len(X),) where the elements sum to 1.
    n : int
        The block length of transmitted sequences.
    M : int
        The dimensionality of the semantic space.
    U : list
        List of task alphabet symbols.
    func_dist : dict
        Function mapping (X,U) pairs to distortion values, in the form of a 
        dictionary with keys (x,u), x in X and u in U, and values as floats.
        Every possible pair must be included in the dictionary.
    rate : int
        The rate of the technical code in bits/semantic representation.
    num_sums : int
        The number of Monte Carlo simulations to run.
    kwargs
        Additional keyword arguments. Recognized arguments include:
        - tol (Float): Tolerance for Lloyd's algorthm. See coding.lloyds_alg
        - max_iters (int): Max iterations of Lloyd's alg. See coding.lloyds_alg
        - min_iters (int): Warmup for Lloyd's alg. See coding.lloyds_alg
    
    Returns
    -------
    avg_sem_dist : float
        The average semantic distortion over the simulations.
    avg_fun_dist : float
        The average functional distortion over the simulations.
    avg_discrp : float
        The average distortion discrepancy over the simulations.
    """
    N = len(X)

    e_s, Z = cd.get_semantic_encoder(X, M, n)

    vnoi, Z_hat = cd.lloyds_alg(Z, p_x, sem_dist, rate, verbose, **kwargs)
    e_t, g_t = cd.tech_code_from_lloyd(vnoi, n)

    g_s = cd.get_semantic_decoder(Z_hat, U, n)

    code = {'e_s': e_s, 'e_t': e_t, 'g_t': g_t, 'g_s': g_s}
    if verbose: print_code(code, Z, Z_hat)

    if verbose: print('\nBeginning Lloyd-random simulations...')
    cum_sem_dist, cum_fun_dist, cum_discrep = 0, 0, 0
    for i in range(num_sims):
        if n == 1:
            x = X[choose(N, p=p_x)]
        else:
            x = tuple(X[choose(N, p=p_x)] for _ in range(n))
        uh, z_id, zh_id = sim_comm(x, code)
        if n == 1:
            d_s = sem_dist(Z[z_id], Z_hat[zh_id])
            d_f = func_dist[(x, uh)]
        if n > 1:
            xs, uhs, z_ids, zh_ids = x, uh, z_id, zh_id
            zs = tuple([Z[z_id] for z_id in z_ids])
            zhs = tuple([Z_hat[zh_id] for zh_id in zh_ids])
            d_s = sem_dist(zs, zhs, n)
            d_f = 1/n*sum(func_dist[(x,uh)] for x, uh in zip(xs, uhs))
        cum_sem_dist += d_s
        cum_fun_dist += d_f
        cum_discrep += (d_s - d_f)**2
    if verbose: print('Done!')

    avg_sem_dist = cum_sem_dist/num_sims
    avg_fun_dist = cum_fun_dist/num_sims
    avg_discrp = cum_discrep/num_sims

    return avg_sem_dist, avg_fun_dist, avg_discrp

# ------------------------------------------------------------------------------

def simulate_system(
        X: list, p_x: np.ndarray, n: int, M: int, U: list, func_dist: dict, 
        rate: int, dec_type: str, num_sims: int, verbose: bool = False, 
        sem_enc_seed: int = None, lloyd_seed: int = None, 
        sem_dec_seed: int = None, **kwargs):
    """
    Simulates a naive technical code that randomly assigns codewords as members
    of the source alphabet, and a semantic decoder that randomly maps semantic
    codewords to members of the task alphabet.

    Parameters
    ----------
    X : list
        X : list[any]
        List of inputs alphabet symbols.
    p_x : np.ndarray
        The probability distribution of the source.
        NumPy array with shape (len(X),) where the elements sum to 1.
    n : int
        The block length of transmitted sequences.
    M : int
        The dimensionality of the semantic space.
    U : list
        List of task alphabet symbols.
    func_dist : dict
        Function mapping (X,U) pairs to distortion values, in the form of a 
        dictionary with keys (x,u), x in X and u in U, and values as floats.
        Every possible pair must be included in the dictionary.
    rate : int
        The rate of the technical code in bits/semantic representation.
    dec_type : str
        The type of semantic decoder to implement in the simulations.
        Options are "min_discrepancy" or "random".
    num_sums : int
        The number of Monte Carlo simulations to run.
    verbose : bool (optional)
        Prints verbose output when True. Default is False.
    sem_enc_seed : int (optional)
        If specified, sets the seed before generating the semantic encoder.
        If None, a random seed is used. Default is None.
    lloyd_seed : int (optional)
        If specified, sets the seed before implementing Lloyd's algorithm.
        If None, a random see is used. Default is None.
    sem_dec_seed : int (optional)
        If specified, sets the seed before generating the random sem. decoder.
        If None, a random seed is used. Default is None.
    kwargs
        Additional keyword arguments. Recognized arguments include:
        - tol (Float): Tolerance for Lloyd's algorthm. See coding.lloyds_alg
        - max_iters (int): Max iterations of Lloyd's alg. See coding.lloyds_alg
        - min_iters (int): Warmup for Lloyd's alg. See coding.lloyds_alg
    
    Returns
    -------
    avg_sem_dist : float
        The average semantic distortion over the simulations.
    avg_fun_dist : float
        The average functional distortion over the simulations.
    avg_discrp : float
        The average distortion discrepancy over the simulations.
    """
    N = len(X)

    X_seqs = extend_list(X, n)
    U_seqs = extend_list(U, n)

    p_x_seqs = []
    for x_seq in X_seqs:
        p_x_seq = 1.0
        for x in x_seq:
            p_x_seq *= p_x[X.index(x)]
        p_x_seqs.append(p_x_seq)

    func_dist_seqs = {}
    for x_seq in X_seqs:
        for u_seq in U_seqs:
            func_dist_seq = 0
            for x, u in zip(x_seq, u_seq):
                func_dist_seq += func_dist[(x, u)]
            func_dist_seqs[(x_seq, u_seq)] = func_dist_seq/n
    
    if sem_enc_seed is not None:
        random_state = np.random.get_state()
        np.random.seed(sem_enc_seed)
        e_s, Z = cd.get_semantic_encoder(X, M, n)
        np.random.set_state(random_state)
    else:
        e_s, Z = cd.get_semantic_encoder(X, M, n)

    if lloyd_seed is not None:
        random_state = np.random.get_state()
        np.random.seed(lloyd_seed)
        vnoi, Z_hat = cd.lloyds_alg(Z, p_x, sem_dist, rate, verbose, **kwargs)
        np.random.set_state(random_state)
    else:
        vnoi, Z_hat = cd.lloyds_alg(Z, p_x, sem_dist, rate, verbose, **kwargs)

    e_t, g_t = cd.tech_code_from_lloyd(vnoi, n)

    if dec_type == 'min_discrepancy':
        if n == 1:
            g_s = cd.min_discrp_sem_dec(X, p_x, n, Z, Z_hat, U, e_s, e_t, g_t, 
                                        func_dist)
        elif n > 1:
            g_s = cd.min_discrp_sem_dec(X_seqs, p_x_seqs, n, Z, Z_hat, U_seqs, 
                                        e_s, e_t, g_t, func_dist_seqs)
    elif dec_type == 'random':
        random_state = np.random.get_state()
        np.random.seed(sem_dec_seed)
        g_s = cd.get_semantic_decoder(Z_hat, U, n)
        np.random.set_state(random_state)


    code = {'e_s': e_s, 'e_t': e_t, 'g_t': g_t, 'g_s': g_s}
    if verbose: print_code(code, Z, Z_hat)

    if verbose: print('\nBeginning simulations...')
    cum_sem_dist, cum_fun_dist, cum_discrep = 0, 0, 0
    for i in range(num_sims):
        if n == 1:
            x = X[choose(N, p=p_x)]
        else:
            x = tuple([X[choose(N, p=p_x)] for _ in range(n)])
        uh, z_id, zh_id = sim_comm(x, code)
        if n == 1:
            d_s = sem_dist(Z[z_id], Z_hat[zh_id])
            d_f = func_dist[(x, uh)]
        if n > 1:
            xs, uhs, z_ids, zh_ids = x, uh, z_id, zh_id
            zs = tuple([Z[z_id] for z_id in z_ids])
            zhs = tuple([Z_hat[zh_id] for zh_id in zh_ids])
            d_s = sem_dist(zs, zhs, n)
            d_f = 1/n*sum(func_dist[(x,uh)] for x, uh in zip(xs, uhs))
        cum_sem_dist += d_s
        cum_fun_dist += d_f
        cum_discrep += (d_s - d_f)**2
        if (i+1) % num_sims/100 == 0:
            print(f'\r{i+1}/{num_sims}', end='')
    if verbose: print('\nDone!')

    avg_sem_dist = cum_sem_dist/num_sims
    avg_fun_dist = cum_fun_dist/num_sims
    avg_discrp = cum_discrep/num_sims

    return avg_sem_dist, avg_fun_dist, avg_discrp

# ------------------------------------------------------------------------------