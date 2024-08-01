"""
This script looks at implementing a simple case study example for the semantic
fucntional rate distortion theory.
"""

# ------------------------------------------------------------------------------
# imports

import numpy as np

# ------------------------------------------------------------------------------

def main():

    np.random.seed(37)
    
    X_set = [x for x in range(4)]           # -> [0, 1, 2, 3]
    px = [1/4 for _ in range(4)]            # -> [1/4, 1/4, 1/4, 1/4]

    U_set = [u for u in range(2)]           # -> [0, 1]

    # df[i][j] = 1 if (X[i] mod 2) != U[j], 0 o.w.
    df = [[1 if x % 2 != u else 0 for u in U_set] for x in X_set]

    rnd = np.random.normal
    Z_set = [(rnd(), rnd()) for _ in range(4)]

    # define functions as dictionaries
    e_s = {x: z for x, z in zip(X_set, Z_set)}
    e_t = {z: i for z, i in zip(Z_set, range(4))}
    g_t = {i: z_hat for i, z_hat in zip(range(4), Z_set)}
    g_s = {z_hat: x%2 for z_hat, x in zip(Z_set, X_set)}

    # get elements for optimizing parameters
    d_vec = np.array([df[x][g_s[g_t[e_t[e_s[x]]]]] for x in X_set]).reshape((4,1))
    Z_mat = np.zeros((4,2))
    for i, x in enumerate(X_set):
        for j in range(2):
            Z_mat[i,j] = (e_s[x][j] - g_t[e_t[e_s[x]]][j])**2
    P_mat = np.diag(px)

    breakpoint()



# ------------------------------------------------------------------------------

if __name__ == '__main__':
    main()

# ------------------------------------------------------------------------------