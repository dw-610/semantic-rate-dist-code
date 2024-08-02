"""
This module contains functions that perform some argument-checking functions
common in other modules.
"""

# ------------------------------------------------------------------------------

import numpy as np

# ------------------------------------------------------------------------------

def check_marginal_dist(arr, arr_name):
    """Check if a NumPy array is a valid row vector marginal distribution."""
    if len(arr.shape) != 2:
        raise ValueError(f'{arr_name} should be 2D (1xN)')
    if arr.shape[0] != 1:
        raise ValueError(f'{arr_name} should be a row vector (1xN)')
    if not np.isclose(np.sum(arr), 1.0):
        raise ValueError(f'{arr_name} elements should sum to 1.')
    if not np.all(arr >= 0.0):
        raise ValueError(f'{arr_name} elements should be nonnegative.')
    
# ------------------------------------------------------------------------------

def check_conditional_dist(arr, arr_name):
    """Checks if a NumPy array is a valid conditional distribution."""
    if len(arr.shape) != 2:
        raise ValueError(f'{arr_name} should be 2D (MxN)')
    if not np.all(np.isclose(np.sum(arr, axis=0), 1.0)):
        raise ValueError(f'{arr_name} columns should sum to 1.')
    
# ------------------------------------------------------------------------------

def check_joint_dist(arr, arr_name):
    """Checks if a NumPy array is a valid joint distribution."""
    if len(arr.shape) != 2:
        raise ValueError(f'{arr_name} should be 2D (NxM)')
    if not np.isclose(np.sum(arr), 1.0):
        raise ValueError(f'{arr_name} elements should sum to 1.')
    if not np.all(arr >= 0.0):
        raise ValueError(f'{arr_name} elements should be nonnegative.')
    
# ------------------------------------------------------------------------------

def check_distortion_mat(arr, arr_name):
    """Checks if a NumPy array is a valid distortion matrix."""
    if len(arr.shape) != 2:
        raise ValueError(f'{arr_name} should be 2D (NxM)')
    if not np.all(arr >= 0.0):
        raise ValueError(f'{arr_name} elements should be nonnegative.')
    
# ------------------------------------------------------------------------------
