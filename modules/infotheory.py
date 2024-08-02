"""
This module contains code for performing various information-theoretic tasks.
"""

# ------------------------------------------------------------------------------
# imports 

import numpy as np
import inspect as ins

from . import checks
from .utils import variable_and_name as var_and_name

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
    checks.check_marginal_dist(*var_and_name(p_x))
    checks.check_conditional_dist(*var_and_name(p_y_given_x))
    return (p_y_given_x*p_x).T

# ------------------------------------------------------------------------------

def get_marginal_dists(p_x_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
        1xN-dim and 1xM-dim 2D vectors, respectively.
    """
    checks.check_joint_dist(*var_and_name(p_x_y))
    N, M = p_x_y.shape
    p_x = np.sum(p_x_y, axis=1).reshape((1,N))
    p_y = np.sum(p_x_y, axis=0).reshape((1,M))
    return p_x, p_y

# ------------------------------------------------------------------------------

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
    checks.check_joint_dist(*var_and_name(p_x_y))
    p_x, p_y = get_marginal_dists(p_x_y)
    I_X_Y = 0
    for i in range(len(p_x)):
        for j in range(len(p_y)):
            if p_x_y[i,j] == 0:
                pass
            else:
                I_X_Y += p_x_y[i,j] * np.log2(p_x_y[i,j]/p_x[i]/p_y[j])
    return float(I_X_Y)

# ------------------------------------------------------------------------------

def get_entropy(p_x: np.ndarray) -> float:
    """
    Computes the entropy H(X) of a source X ~ p(x).

    Parameters
    ----------
    p_x : np.ndarray
        The marginal distribution of X, specified as a 1xN 2D array.

    Returns
    -------
    float
        The entropy H(X).
    """ 
    checks.check_marginal_dist(*var_and_name(p_x))
    return float(-p_x @ np.log2(p_x).T)

# ------------------------------------------------------------------------------

def get_joint_entropy(p_x_y: np.ndarray) -> float:
    """
    Computes the joint entropy H(X,Y) of a source X, Y ~ p(x,y).

    Parameters
    ----------
    p_x_y : np.ndarray
        The joint distribution p(x,y), specified as a 2D NxM matrix. The [i,j]th 
        entry corresponds to p(x_i, y_j).

    Returns
    -------
    float
        The entropy H(X,Y).
    """ 
    checks.check_joint_dist(*var_and_name(p_x_y))
    return float(-p_x_y.flatten() @ np.log2(p_x_y.flatten()))

# ------------------------------------------------------------------------------

def get_expected_distortion(p_x_y: np.ndarray, d_x_y: np.ndarray) -> float:
    """
    Computes the expected distortion givena a joint distribution p(x,y) and a 
    distortion function d(x,y). These arrays should be the same shape.

    Parameters
    ----------
    p_x_y : np.ndarray
        The joint distribution p(x,y), specified as a NxM matrix. The [i,j]th 
        entry corresponds to p(x_i, y_j).
    d_x_y : np.ndarray
        The distortion values for (x, \hat{x}) pairs, specified as a NxM matrix. 
        The entry d_x_y[i, j] gives the distortion d(x_i, \hat{x}_j).

    Returns
    -------
    float
        The expected distortion E[d(x,y)].
    """
    checks.check_joint_dist(*var_and_name(p_x_y))
    checks.check_distortion_mat(*var_and_name(d_x_y))
    return np.sum(p_x_y*d_x_y)

# ------------------------------------------------------------------------------

def blahut_arimoto(p_x: np.ndarray, d_x_y: np.ndarray, beta: float = 1.0,
                   tol: float = 1e-3, max_iter: int = 100) -> np.ndarray:
    """
    Implements the Blahut-Arimoto algorithm for finding the optimal conditional
    distribution that minimizes mutual info given a distortion constraint.
    Further information on 
    [Wikipedia](https://en.wikipedia.org/wiki/Blahut-Arimoto_algorithm).

    This implementation assumes the size of the source and codebook alphabets
    are the same (i.e., conditional dist matrix is square).

    The parameter beta acts as the distortion contraint in a sense. A greater
    value of beta -> less distortion and greater rate. A lower value of beta ->
    greater distortion and lower rate.

    Think of this algorithm as returning a point on the R(D) curve, where beta
    "slides" you along the curves.

    Parameters
    ----------
    p_x : np.ndarray
        The marginal distribution of x, specified as a 1xN vector.
        Entries should sum to 1.
    d_x_y : np.ndarray
        The distortion values for (x, \hat{x}) pairs. The entry d_x_y[i, j]
        gives the distortion d(x_i, \hat{x}_j).
    beta : float
        Parameter that controls "where" on the R(D) you land. Greater value of 
        beta -> less distortion and greater rate. A lower value of beta -> 
        greater distortion and lower rate.
    tol : float
        The tolerance for convergence of the algorithm, which is measured as the
        Frobenius norm of the difference in conditional distribution matrices
        from one iteration to the next.
    max_iter : int
        The maximum number of steps that will be taken in the algorithm.

    Returns
    -------
    np.ndarray
        The found conditional distribution of y given x, specified as a MxN 
        matrix, where the [i,j]th entry denotes p(y_i | x_j).
    """
    checks.check_marginal_dist(*var_and_name(p_x))
    checks.check_distortion_mat(*var_and_name(d_x_y))

    p_x = p_x.flatten()
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

    if iter == max_iter:
        print(f'Blahut-Arimoto: Did not converge in {max_iter} iterations.')
    else:
        print(f'Blahut-Arimoto: Converged in {iter} iterations.')

    return p_y_given_x

# ------------------------------------------------------------------------------