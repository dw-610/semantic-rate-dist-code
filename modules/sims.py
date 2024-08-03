"""
This module contains high-level simulation routines.
"""

# ------------------------------------------------------------------------------

import numpy as np
from numpy.random import choice as choose

from . import coding as cd
from .coding import semantic_distortion as sem_dist
from .utils import print_code

# ------------------------------------------------------------------------------

def sim_comm(x, code: dict) -> tuple:
    """
    Simulates a single communication sequence, from the generation of a source
    symbol x to the output of the estimated task variable u_hat.
    
    Parameters
    ----------
    x : Any (element of the source alphabet X)
        Source symbol that is the input to the system.
    codes : dict
        A dictionary containing keyword pairs for the code functions. Should
        have the following keys/value pairs:
        - 'e_s': dict (semantic encoder)
        - 'e_t': dict (technical encoder)
        - 'g_t': dict (technical decoder)
        - 'g_s': dict (semantic decoder)
    
    Returns
    -------
    u_hat : Any
        The output of the communication chain for the input x.
    z_id : int
        ID of the semantic representation for x.
    zh_id : int
        ID of the recovered semantic representation.
    """
    e_s, e_t, g_t, g_s = [code[k] for k in ('e_s', 'e_t', 'g_t', 'g_s')]
    
    z_id    = e_s[x]
    cw      = e_t[z_id]
    zh_id   = g_t[cw]
    uh      = g_s[zh_id]

    return uh, z_id, zh_id

# ------------------------------------------------------------------------------

def naive_technical_random_decoder(
        X: list, p_x: np.ndarray, Z: dict, U: list, func_dist: dict, rate: int, 
        num_sims: int, verbose = False):
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
    Z : dict[int] -> np.ndarray
        Dictionary mapping ID's (int) to semantic reps (NumPy arrays).
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
    N = len(X)

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
        X: list, p_x: np.ndarray, Z: dict, U: list, func_dist: dict, rate: int, 
        num_sims: int, verbose = False, **kwargs):
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
    Z : dict[int] -> np.ndarray
        Dictionary mapping ID's (int) to semantic reps (NumPy arrays).
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

    vnoi, Zh_set = cd.lloyds_alg(Z, p_x, sem_dist, rate, verbose, **kwargs)
    code = {
        'e_s': cd.get_semantic_encoder(X, Z),
        **dict(zip(['e_t', 'g_t'], 
                   cd.tech_code_from_lloyd(vnoi))),
        'g_s': cd.get_semantic_decoder(Zh_set, U)
    }
    if verbose: print_code(code, Z, Zh_set)

    if verbose: print('\nBeginning Lloyd-random simulations...')
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

def lloyd_technical_optimal_decoder(
        X: list, p_x: np.ndarray, Z: dict, U: list, func_dist: dict, rate: int, 
        num_sims: int, verbose = False, **kwargs):
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
    Z : dict[int] -> np.ndarray
        Dictionary mapping ID's (int) to semantic reps (NumPy arrays).
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
    
    vnoi, Z_hat = cd.lloyds_alg(Z, p_x, sem_dist, rate, verbose, **kwargs)
    e_t, g_t = cd.tech_code_from_lloyd(vnoi)
    code = {
        'e_s': cd.get_semantic_encoder(X, Z),
        'e_t': e_t,
        'g_t': g_t,
        'g_s': cd.min_discrp_sem_dec(X, p_x, Z, Z_hat, U, e_t, g_t, func_dist)
    }
    if verbose: print_code(code, Z, Z_hat)

    if verbose: print('\nBeginning Lloyd-optimal simulations...')
    cum_sem_dist, cum_fun_dist, cum_discrep = 0, 0, 0
    for i in range(num_sims):
        x = X[choose(N, p=p_x)]
        uh, z_id, zh_id = sim_comm(x, code)
        try:
            d_s = sem_dist(Z[z_id], Z_hat[zh_id])
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