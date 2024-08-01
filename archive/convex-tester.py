"""
This script tests whether a function is convex or not.

A function is convex if f(ax + (1-a)y) <= af(x) + (1-a)f(y) for all x, y in the domain
"""

# ------------------------------------------------------------------------------
# imports

import numpy as np

# ------------------------------------------------------------------------------

def func(a, p, Z, u, delta):
    """a: Mx1, p: Nx1, Z: NxM, u: scalar, delta: Nx1"""
    df = (Z @ a - u)**2
    diff = (df - delta)**2
    return (p.T @ diff).squeeze()

def convex_func(a, Z):
    return a.T @ (Z.T @ Z) @ a

def ip_gen(M):
    while True:
        yield np.random.normal(size=(M,1))

def test_convexity(num, func, ip_gen, **kwargs):
    for _ in range(num):
        ip1 = next(ip_gen)
        ip2 = next(ip_gen)

        lam = np.random.random()

        combo_ip = lam*ip1 + (1-lam)*ip2
        lhs = func(combo_ip, **kwargs)

        f1 = func(ip1, **kwargs)
        f2 = func(ip2, **kwargs)
        rhs = lam*f1 + (1-lam)*f2

        if lhs > rhs:
            print(f'Not convex! RHS greater than LHS by {lhs-rhs}')
            return
    print('Did not disprove convexity.')

# ------------------------------------------------------------------------------

def main():

    NUM = 1000000

    N = 10
    M = 5
    
    p = np.random.random(size=(N,1))
    p /= np.sum(p)
    Z = np.random.normal(size=(N, M))
    u = np.random.random()
    delta = np.random.random(size=(N,1))

    gen = ip_gen(M)

    test_convexity(NUM, func, gen, p=p, Z=Z, u=u, delta=delta)
    # test_convexity(NUM, convex_func, gen, Z=Z)

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    main()

# ------------------------------------------------------------------------------