"""
This module contains routines that handle the coding functionality for the 
semantic functional communication system.
"""

# ------------------------------------------------------------------------------
# imports

from typing import Callable
import numpy as np
from numpy.random import randint
from numpy.random import random_sample as rand
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------

def extend_list_to_seq(single_list: list, n: int):
    """Extends the elements of a `list` to length-n `tuple` objects."""
    n_list = [()]
    for i in range(n):
        new_list = []
        for x_seq in n_list:
            for x in single_list:
                new_list.append((*x_seq, x))
        n_list = new_list
    return n_list

# ------------------------------------------------------------------------------

def extend_dict_to_seq(single_dict: dict, n: int):
    """Extends the key-value pairs of a dictionary to length-n sequences."""
    n_dict = {(): ()}
    for i in range(n):
        n_dict_new = {}
        for k_seq, v_seq in n_dict.items():
            for k, v in single_dict.items():
                n_dict_new[(*k_seq, k)] = (*v_seq, v)
        n_dict = n_dict_new
    return n_dict

# ------------------------------------------------------------------------------

def get_semantic_encoder(X: list, M: int, n: int = 1) -> tuple[dict, dict]:
    """
    Returns a semantic encoding function for the source alphabet X.

    If n=1 (default), the returned encoder takes elements of X as inputs. If
    n>1, the encoder takes n-length `tuple` objects of elements of X as inputs.

    Right now, just randomly samples points in M-dimensional space from i.i.d.
    N(0, 1) distributions.

    Parameters
    ----------
    X : list[any]
        List of input alphabet symbols or sequences. 
    M : int
        The dimensionality of the semantic space.
    n : int (optional)
        The block length of transmitted sequences.

    Returns
    -------
    e_s : dict[any or tuple[any]] -> int or tuple[int]
        Dictionary mapping the inputs symbols (or sequences thereof) to the 
        ID's of semantic representation (or sequences thereof).
    Z : dict[int] -> np.ndarray
        Dictionary mapping ID's (int) to semantic reps (NumPy arrays).
    """
    if not isinstance(n, int): raise ValueError("n must be an integer")

    N= len(X)
    get_rep = lambda M : np.random.normal(size=(M,))
    
    if n == 1:
        Z = {z_id: get_rep(M) for z_id in range(N)}
        g_s = {x: z_id for z_id, x in enumerate(X)}
    elif n > 1:
        Z = {z_id: get_rep(M) for z_id in range(N)}
        g_s_single = {x: z_id for z_id, x in enumerate(X)}
        g_s = extend_dict_to_seq(g_s_single, n)
    else:
        raise ValueError("n must be an integer >= 1")
    return g_s, Z

# ------------------------------------------------------------------------------

def get_technical_code(Z: dict, Z_hat: dict, R: int) -> tuple[dict, dict]:
    """
    DEPRECATED - use tech_code_from_lloyd instead

    Returns rate R technical encoding/decoding functions for semantic alphabet 
    Z and recovered alphabet Z_hat.

    This is a naive techincal code that just randomly assigns codewords to
    members of the source alphabets, in the case of compression.

    Three possibilities (N = len(Z)):
      - 2**R = N -> one to one coding
      - 2**R > N -> uniquely decodable -> randomly assign cw's to Z's
      - 2**R < N -> nonuniquely decodable -> enumerate cws (repeating)

    Parameters
    ----------
    Z : dict[int] = np.ndarray
        Dictionary mapping ID's (int) to semantic reps (NumPy arrays).
    Z_hat : dict[int] = np.ndarray
        Dictionary mapping ID's (int) to recovered semantic reps (NumPy arrays).
    R : int
        The rate of the code -> semantic reps mapped to 2^R-bit strings.

    Returns
    -------
    e_t : dict[int] -> int
        Technical encoder, mapping ID's (int) of semantic representations to
        ID's (int) of codewords.
    g_t : dict[int] -> int
        Technical decoder, mapping ID's (int) of codewords to ID's (int) of 
        semantic representations.
    """
    print("\n**WARNING** get_technical_code is deprecated.",
          "Use tech_code_from_lloyd.\n")
    return None
    N = len(Z)
    cw_set = [i for i in range(2**R)]
    if 2**R == N:
        e_t = {z_id: cw for z_id, cw in zip(Z, cw_set)}
        g_t = {cw: zh_id for cw, zh_id in zip(cw_set, Z_hat)}
    if 2**R > N:
        cws = cw_set[:N]
        e_t = {z_id: cw for z_id, cw in zip(Z, cws)}
        g_t = {cw: zh_id for cw, zh_id in zip(cws, Z_hat)}
    if 2**R < N:
        cws = []
        while len(cws) < N:
            if N - len(cws) < 2**R:
                cws.extend(cw_set[:N-len(cws)])
            else:
                cws.extend(cw_set)
        e_t = {z_id: cw for z_id, cw in zip(Z, cws)}
        g_t = {cw: zh_id for cw, zh_id in zip(cws, Z_hat)}
    return e_t, g_t

# ------------------------------------------------------------------------------

def tech_code_from_lloyd(voronoi: dict, n: int) -> tuple[dict, dict]:
    """
    Returns rate R technical encoding/decoding functions resulting from running
    Lloyd's algorithm.

    Note that the "technical decoder" is sort of superfluous in this scenaio
    since we are not actually mapping back to the source alphabet for Z. This
    function will just be the identity mapping, and the recovered semantic
    representation can be accessed using the ID->rep dictionary: Z_hat[g_t[cw]]

    Parameters
    ----------
    voronoi : dict
        Output of the lloyds_alg function.
        A dictionary where the keys correspond to the ID's of the codewords,
        and the values are lists containing the ID's of the semantic
        representations that map to the codeword.
    n : int
        The block length of transmitted sequences.
        
    Returns
    -------
    e_t : dict[int] -> int
        Technical encoder, mapping ID's (int) of semantic representations (or 
        sequences thereof) to ID's (int) of codewords (or sequences thereof).
    g_t : dict[int] -> int
        Technical decoder, mapping ID's (int) of codewords (or sequences 
        thereof) to ID's (int) of semantic representation codewords (or 
        sequences thereof).
    """
    e_t_single = {}
    g_t_single = {}
    for zh_id in voronoi:
        for z_id in voronoi[zh_id]:
            e_t_single[z_id] = zh_id
        g_t_single[zh_id] = zh_id

    if n == 1:
        e_t, g_t = e_t_single, g_t_single
    elif n > 1:
        e_t, g_t = [extend_dict_to_seq(d,n) for d in [e_t_single, g_t_single]]
    else:
        raise ValueError("n must be an integer >= 1")
    return e_t, g_t

# ------------------------------------------------------------------------------

def rand_semantic_decoder(Z_hat: dict, U: list, n: int) -> dict:
    """
    Returns a semantic decoding function mapping the recovered semantic reps
    \hat{Z} to the task alphabet U. Just randomly chooses elements of U to map 
    to - used as a naive baseline in our experiments.

    Parameters
    ----------
    Z_hat : dict[int] = np.ndarray
        Dictionary mapping ID's (int) to recovered semantic reps (NumPy arrays).
    U : list
        List of task alphabet symbols.
    n : int (optional)
        The block length of transmitted sequences.

    Returns
    -------
    g_s : dict
        Dictionary mapping the Z_hat elements to task alphabet symbols.
    """
    g_s_single = {zh_id: U[randint(len(U))] for zh_id in Z_hat}
    if n == 1:
        g_s = g_s_single
    elif n > 1:
        g_s = extend_dict_to_seq(g_s_single, n)
    else:
        raise ValueError("n must be an integer >= 1")
    return g_s

# ------------------------------------------------------------------------------

def min_discrp_sem_dec(X, p_x, n, Z, Z_hat, U, e_s, e_t, g_t, 
                       func_dist) -> dict:
    """
    Returns the semantic decoder which minimizes average distortion discrepancy.

    Accomplishes this with a simple process:
    - If a semantic codeword corresponds to only one source symbol, map it to 
      the task symbol which yield the distortion closest the semantic distortion
    - Otherwise, map to the task symbol that minimizes the weighted sum of the
      discrepancies over the corresponding source symbols (weights are probs)

    Parameters
    ----------
    X : list
        X : list[any]
        List of input alphabet symbols (or sequences of). If n > 1, then these
        should be n-length tuples of source symbols.
    p_x : np.ndarray
        The probability distribution of the source symbols (or sequences).
        NumPy array with shape (len(X),) where the elements sum to 1.
    n : int (optional)
        The block length of transmitted sequences.
    Z : dict[int] -> np.ndarray
        Dictionary mapping ID's (int) to semantic reps (NumPy arrays).
    Z_hat : dict[int] = np.ndarray
        Dictionary mapping ID's (int) to recovered semantic reps (NumPy arrays).
    U : list
        List of task alphabet symbols (or sequences of). The n > 1, then these
        should be n-length tuples of task symbols.
    e_s : dict[any or tuple[any]] -> int or tuple[int]
        Dictionary mapping the inputs symbols (or sequences thereof) to the 
        ID's of semantic representation (or sequences thereof).
    e_t : dict[int] -> int
        Technical encoder, mapping ID's (int) of semantic representations to
        ID's (int) of codewords.
    g_t : dict[int] -> int
        Technical decoder, mapping ID's (int) of codewords to ID's (int) of 
        semantic representations.
    func_dist : dict
        Function mapping (X,U) pairs to distortion values, in the form of a 
        dictionary with keys (x,u), x in X and u in U, and values as floats.
        Every possible pair must be included in the dictionary.

    Returns
    -------
    g_s : dict
        Dictionary mapping the Z_hat elements to task alphabet symbols.    
    """
    delta = semantic_distortion

    N, K, R = len(X), len(U), int(np.log2(len(Z_hat)))
    d_np = np.zeros((N,K))
    for i, x in enumerate(X):
        for j, u in enumerate(U):
            d_np[i,j] = func_dist[(x,u)]
    
    d_s_np = np.zeros((N,))
    for i in range(N):
        if n == 1:
            d_s_np[i] = delta(Z[e_s[X[i]]], Z_hat[g_t[e_t[e_s[X[i]]]]])
        else:
            z_seq = tuple([Z[z_id] for z_id in e_s[X[i]]])
            zh_seq = tuple([Z_hat[zh_id] for zh_id in g_t[e_t[e_s[X[i]]]]])
            d_s_np[i] = delta(z_seq, zh_seq, n)

    rev_e_s = {z_seq: x_seq for x_seq, z_seq in e_s.items()}

    g_s = {}
    for zh in g_t.values():
        x_ids = []
        # determine if Z_hat[i] corresponds to more than one source symbol/seq
        for z_id, zh_id in e_t.items():
            if zh_id == zh:
                x_ids.append(rev_e_s[z_id])
        # if only one, set g_s(i) equal to u that minimizes (d - delta)^2
        if len(x_ids) == 1:
            x_seq_id = X.index(x_ids[0])
            g_s[zh] = U[np.argmin((d_np[x_seq_id,:] - d_s_np[x_seq_id])**2)]
        # if more than one, need to choose the U that minimizes discrepancy
        # accross these source symbols, weighted by their probabilities
        if len(x_ids) > 1:
            min_cum, min_id = 1e9, None
            for u_id in range(K):
                cum = 0
                for x_id in x_ids:
                    x_seq_id = X.index(x_id)
                    diff_sq = (d_np[x_seq_id, u_id] - d_s_np[x_seq_id])**2    
                    cum += p_x[x_seq_id] * diff_sq
                if cum < min_cum:
                    min_cum = cum
                    min_id = u_id
            g_s[zh] = U[min_id]
    return g_s    

# ------------------------------------------------------------------------------

def min_distortion_sem_dec(X, p_x, n, Z, Z_hat, U, e_s, e_t, g_t, 
                           func_dist) -> dict:
    """
    Returns the semantic decoder which minimizes average functional distortion.
    Used as an "oracle" decoder in our experiments.

    Accomplishes this with a simple process:
    - If a semantic codeword corresponds to only one source symbol, map it to 
      the task symbol which yield the distortion closest the semantic distortion
    - Otherwise, map to the task symbol that minimizes the weighted sum of the
      discrepancies over the corresponding source symbols (weights are probs)

    Parameters
    ----------
    X : list
        X : list[any]
        List of input alphabet symbols (or sequences of). If n > 1, then these
        should be n-length tuples of source symbols.
    p_x : np.ndarray
        The probability distribution of the source symbols (or sequences).
        NumPy array with shape (len(X),) where the elements sum to 1.
    n : int (optional)
        The block length of transmitted sequences.
    Z : dict[int] -> np.ndarray
        Dictionary mapping ID's (int) to semantic reps (NumPy arrays).
    Z_hat : dict[int] = np.ndarray
        Dictionary mapping ID's (int) to recovered semantic reps (NumPy arrays).
    U : list
        List of task alphabet symbols (or sequences of). The n > 1, then these
        should be n-length tuples of task symbols.
    e_s : dict[any or tuple[any]] -> int or tuple[int]
        Dictionary mapping the inputs symbols (or sequences thereof) to the 
        ID's of semantic representation (or sequences thereof).
    e_t : dict[int] -> int
        Technical encoder, mapping ID's (int) of semantic representations to
        ID's (int) of codewords.
    g_t : dict[int] -> int
        Technical decoder, mapping ID's (int) of codewords to ID's (int) of 
        semantic representations.
    func_dist : dict
        Function mapping (X,U) pairs to distortion values, in the form of a 
        dictionary with keys (x,u), x in X and u in U, and values as floats.
        Every possible pair must be included in the dictionary.

    Returns
    -------
    g_s : dict
        Dictionary mapping the Z_hat elements to task alphabet symbols.    
    """
    delta = semantic_distortion

    N, K, R = len(X), len(U), int(np.log2(len(Z_hat)))
    d_np = np.zeros((N,K))
    for i, x in enumerate(X):
        for j, u in enumerate(U):
            d_np[i,j] = func_dist[(x,u)]
    
    d_s_np = np.zeros((N,))
    for i in range(N):
        if n == 1:
            d_s_np[i] = delta(Z[e_s[X[i]]], Z_hat[g_t[e_t[e_s[X[i]]]]])
        else:
            z_seq = tuple([Z[z_id] for z_id in e_s[X[i]]])
            zh_seq = tuple([Z_hat[zh_id] for zh_id in g_t[e_t[e_s[X[i]]]]])
            d_s_np[i] = delta(z_seq, zh_seq, n)

    rev_e_s = {z_seq: x_seq for x_seq, z_seq in e_s.items()}

    g_s = {}
    for zh in g_t.values():
        x_ids = []
        # determine if Z_hat[i] corresponds to more than one source symbol/seq
        for z_id, zh_id in e_t.items():
            if zh_id == zh:
                x_ids.append(rev_e_s[z_id])
        # if only one, set g_s(i) equal to u that minimizes *distortion*
        if len(x_ids) == 1:
            x_seq_id = X.index(x_ids[0])
            g_s[zh] = U[np.argmin(d_np[x_seq_id,:])]
        # if more than one, need to choose the U that minimizes *distortion*
        # accross these source symbols, weighted by their probabilities
        if len(x_ids) > 1:
            min_cum, min_id = 1e9, None
            for u_id in range(K):
                cum = 0
                for x_id in x_ids:
                    x_seq_id = X.index(x_id)
                    distortion = d_np[x_seq_id, u_id]
                    cum += p_x[x_seq_id] * distortion
                if cum < min_cum:
                    min_cum = cum
                    min_id = u_id
            g_s[zh] = U[min_id]
    return g_s    

# ------------------------------------------------------------------------------

def semantic_distortion(z1: np.ndarray, z2: np.ndarray, n: int = 1) -> float:
    """
    Computes semantic distortion between vector (or sequences of) z1 and z2.

    If sequences are passed, the mean of the individual distortions is returned.

    Parameters
    ----------
    z1 : np.ndarray
        A (N,) NumPy array containing coordinates in the conceptual space, OR an
        n-length `tuple` of (N,) NumPy arrays.
    z1 : np.ndarray
        A (N,) NumPy array containing coordinates in the conceptual space, OR an
        n-length `tuple` of (N,) NumPy arrays.
    n : int (optional)
        The block length of transmitted sequences. Default is 1.

    Returns
    -------
    d_s(z1, z2) : float
        The (mean) semantic distortion between (sequences of) vectors z1 and z2.
    """
    if n == 1:
        return np.linalg.norm(z1 - z2)
    elif n > 1:
        cum_dist = 0
        for i in range(n):
            cum_dist += np.linalg.norm(z1[i] - z2[i])
        return cum_dist/n

# ------------------------------------------------------------------------------

def lloyds_alg(Z: dict, p_z: np.ndarray, delta: Callable, R: int, 
               verbose: bool = False, tol: float = 1e-3, max_iters: int = 1e3,
               min_iters: int = 100, plot: bool = False, **kwargs) -> dict:
    """
    This function implements Lloyd's algorithm to determine the optimal
    codewords for a given source alphabet of semantic representations, with
    an added stochastic element to help break out of local optima.

    Parameters
    ----------
    Z : dict[int] = np.ndarray
        Dictionary mapping ID's (int) to semantic reps (NumPy arrays).
    p_z : np.ndarray
        The probability distribution of the semantic representations.
        NumPy array with shape (len(Z),) where the elements sum to 1.
    delta : function
        The semantic distortion function. Should take two vector (z1, z2) and
        return a non-negative real number.
    R : int
        The rate of communication (i.e., bits/represenation).
    tol : float (optional)
        The tolerance used for decided when the algorithm has convered.
        Default is 1e-3.
    max_iters : int (optional)
        The maximum number of iterations the algorithm will run for.
        Default is 1000.
    min_iters : int (optional)
        The minimum number of iterations the algorithm will run for. This acts
        as a sort of "warm up" period for the algorithm to escape local minima.
        Default is 100.
    verbose : bool (optional)
        If True, the algorithm will print some feedback each iteration.
        Default is False.
    plot : bool (optional)
        IF True and M = 2, will show an updating plot of the constellation.
        Default is False.

    Returns
    -------
    vnoi : dict
        A dictionary where the keys correspond to the ID's of the codewords,
        and the values are lists containing the ID's of the semantic
        representations that map to the codeword.
    Zh : dict
        A dictionary mapping the ID's of the codewords to their vector values.
    """
    if len(kwargs) > 0:
        print('Warning: got unexpected keyword arguments for lloyds_alg:',
              *[keyword for keyword in kwargs])
    M = len(Z[0])

    # 1. Initialize the codebook with codewords z_hat (2^R Mx1 real vectors)
    Zh = {i: 4*(rand(size=(M,))-0.5) for i in range(2**R)}

    # 2. Get the Voronoi tessellation - assign each z to the closest z_hat
    vnoi = {i: [] for i in Zh} # vnoi[codeword_id] -> (alphabet ids)
    for z_id, z in Z.items():
        min_d = 1e9
        min_id = None
        for zh_id, zh in Zh.items():
            if delta(zh, z) < min_d:
                min_d = delta(zh, z)
                min_id = zh_id
        vnoi[min_id].append(z_id)

    if plot:
        if M != 2:
            raise Warning("Can only plot with M=2.")
        else:
            plot_2D_semantic_constellation(Z, Zh, vnoi)
            plt.draw()
            plt.pause(0.25)
            plt.clf()
        
    # 3. Get the expected semantic distortion for this assignment
    E_delta = 0
    for zh_id, cell in vnoi.items():
        for z_id in cell:
            E_delta += p_z[z_id]*delta(Z[z_id], Zh[zh_id])

    if verbose: print("Beginning Lloyd's alg...")
    change = 1e9
    iters = 0
    while ((change > tol and iters < max_iters) or iters < min_iters) and E_delta > 0:
        iters += 1
        E_delta_old = E_delta

        # 4. Compute the centroids of each Voronoi region -> new codewords
        # NOTE: this is specific to the squared-euclidean distance. Centroid changes
        # when the notion of distance/divergence changes.
        for zh_id, cell in vnoi.items():
            if len(cell) > 1:
                centroid = np.zeros(M)
                cell_ps = np.array([p_z[i] for i in cell])
                cell_ws = cell_ps / np.sum(cell_ps)
                for i, z_id in enumerate(cell):
                    centroid += cell_ws[i]*Z[z_id]
                Zh[zh_id] = centroid
            elif len(cell) == 1:
                Zh[zh_id] = Z[cell[0]]
            else:
                # if there are no reps assigned to a codeword, sample a new one
                Zh[zh_id] = 4*(rand(size=(M,))-0.5)

        # 5. Recompute the Voronoi tesselation and the expected semantic distortion
        vnoi = {i: [] for i in Zh} # vnoi[codeword_id] -> (alphabet ids)
        for z_id, z in Z.items():
            min_d = 1e9
            min_id = None
            for zh_id, zh in Zh.items():
                if delta(zh, z) < min_d:
                    min_d = delta(zh, z)
                    min_id = zh_id
            vnoi[min_id].append(z_id)

        E_delta = 0
        for zh_id, cell in vnoi.items():
            for z_id in cell:
                E_delta += p_z[z_id]*delta(Z[z_id], Zh[zh_id])

        # 6. Check the change in the expected distortion
        change = np.abs(E_delta - E_delta_old)

        if verbose:
            print(f'({iters}) Old: {E_delta_old:.2f}', f'New: {E_delta:.2f}', 
                f'Change: {change:.8f}', sep=', ')
            
        if plot:
            if M != 2:
                raise Warning("Can only plot with M=2.")
            else:
                plot_2D_semantic_constellation(Z, Zh, vnoi)
                plt.draw()
                plt.pause(0.5)
                plt.clf()
    if verbose: print('Done!')
        
    return vnoi, Zh

# ------------------------------------------------------------------------------

def plot_2D_semantic_constellation(Z: dict, Zh: dict, voronoi):
    """Plots a visualization of the technical code resulting from the GLA."""
    N = len(Z)
    M = len(Z[0])
    R = int(np.log2(len(Zh)))
    Z_np, Zh_np = np.zeros((N,M)), np.zeros((2**R,M))
    for i in range(N):
        Z_np[i,:] = Z[i]
    for i in range(2**R):
        Zh_np[i,:] = Zh[i]

    if M == 2:
        plt.scatter(Z_np[:,0], Z_np[:,1], color='blue', s=20, marker='.',
                    alpha=0.4)
        minx, maxx = 1e9, -1e9
        miny, maxy = 1e9, -1e9
        for i in range(2**R):
            zh = Zh_np[i,:]
            for z_id in voronoi[i]:
                z = Z[z_id]
                xs = [zh[0], z[0]]
                ys = [zh[1], z[1]]
                if z[0] < minx: minx = z[0]
                if z[0] > maxx: maxx = z[0]
                if z[1] < miny: miny = z[1]
                if z[1] > maxy: maxy = z[1]
                plt.plot(xs, ys, color='black', linewidth=0.2, linestyle=':',
                         alpha=1.0)
        plt.scatter(Zh_np[:,0], Zh_np[:,1], color='red', s=20, marker='o',
                    alpha=0.6, zorder=1e9)
        # plt.grid()
        plt.xlim([minx, maxx])
        plt.ylim([miny, maxy])
        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.show()

# ------------------------------------------------------------------------------

if __name__=="__main__":

    # ------------------------------------------------------------
    # # testing Lloyd's algorithm
    # from numpy.random import normal as randn
    # N = 1000
    # M = 2
    # R = 4
    # Z_set = {i: randn(size=(M,)) for i in range(N)}
    # p_z = rand(N); p_z /= np.sum(p_z)
    # plt.figure(figsize=(12,12))
    # vnoi, Zh = lloyds_alg(Z_set, p_z, semantic_distortion, R, verbose=True,
    #                       min_iters=5, plot=False)
    # plot_2D_semantic_constellation(Z_set, Zh, vnoi)
    # # print(Z_set, Zh, vnoi, sep='\n')
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # testing semantic encoder for sequences
    X = [char for char in 'abcde']
    n = 5
    M = 2
    g_s, Z = get_semantic_encoder(X, M, n)
    vnoi, Zh = lloyds_alg(Z)    
    # print('\n', g_s)
    print('\n', Z)
    print('\n', len(g_s))
    print('\n', g_s[('a',)*5])
    # --------------------------------------------------------------
