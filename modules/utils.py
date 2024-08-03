"""
This module contains miscellaneous utility functions.
"""

# ------------------------------------------------------------------------------
# imports

import inspect
import numpy as np

# ------------------------------------------------------------------------------

def variable_and_name(variable):
    """Returns the passed variable and the name of the variable as a str."""
    frame = inspect.currentframe().f_back
    variable_name = None
    for name, value in frame.f_locals.items():
        if value is variable:
            variable_name = name
            break
    return variable, variable_name

# ------------------------------------------------------------------------------

def print_code(code, Z, Z_hat) -> None:
    """Prints out a semantic-functional code in a nice formatted manner."""
    print('\nSemantic Encoder (X -> Z)\n-------------------------')
    for k, v in code['e_s'].items():
        z = np.array2string(Z[v], precision=2, sign='+', separator=', ')
        print(f'  {k} -> {z}')
    print('\nTechnical Encoder (Z -> CW)\n---------------------------')
    for k, v in code['e_t'].items():
        z = np.array2string(Z[k], precision=2, sign='+', separator=', ')
        print(f'  {z} -> {v}')
    print('\nTechnical Decoder (CW -> Z_hat)\n-------------------------------')
    for k, v in code['g_t'].items():
        z_hat = np.array2string(Z_hat[v], precision=2, sign='+', separator=', ')
        print(f'  {k} -> {z_hat}')
    print('\nSemantic Decoder (Z_hat -> U)\n-----------------------------')
    for k, v in code['g_s'].items():
        z_hat = np.array2string(Z_hat[k], precision=2, sign='+', separator=', ')
        print(f'  {z_hat} -> {v}')
    print()

# ------------------------------------------------------------------------------