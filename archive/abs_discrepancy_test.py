"""
This script does some empirical tests to try and disprove the "long sequence
discrepancy is just symbol discrepancy" result.
"""

# ------------------------------------------------------------------------------
# imports

import numpy as np

# ------------------------------------------------------------------------------

def main():

    N = 1000
    M = 100

    K = 10000

    d = np.random.rand(M)
    delta = np.random.rand(M)

    p = np.random.rand(M)
    p /= sum(p)

    # compute the result (abs of expected distortion difference)
    diff = d - delta
    E_diff = p.T @ diff
    abs_E_diff = abs(E_diff)

    # empirically approximate the expectation of the n-sequence abs discrepancy
    sum_abs_diff = 0
    for _ in range(K):
        d_seq = np.random.choice(d, N, True, p)
        delta_seq = np.random.choice(delta, N, True, p)

        d_mean = np.mean(d_seq)
        delta_mean = np.mean(delta_seq)

        sum_abs_diff += abs(d_mean - delta_mean)
    E_abs_diff_n = sum_abs_diff / K

    print(f'Expected distortion E[d]:        {abs_E_diff:.5f}')
    print(f'(Approx) exp. seq. dist. E[d^n]: {E_abs_diff_n:.5f}')

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    main()

# ------------------------------------------------------------------------------