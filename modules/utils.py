"""
This module contains miscellaneous utility functions.
"""

# ------------------------------------------------------------------------------
# imports

import numpy as np

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