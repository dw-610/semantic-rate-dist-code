"""
This script will look at the notion of typicality on some actual data. 

Uses a Monte Carlo simulation to get the empirical probability of the typical 
set of i.i.d. n-sequences of some discrete distribution.
"""

# ------------------------------------------------------------------------------
# imports

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------

def main():
    
    NUM_SYMBOLS     = 10
    SEQ_LENGTHS     = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 
                       8192, 16384, 32768]

    NUM_SIMS        = 10000

    EPSILON       = 0.01

    DIST = np.random.rand(NUM_SYMBOLS)
    DIST = DIST / DIST.sum()

    HX = -np.sum(DIST * np.log(DIST))

    eps_typicals = []
    for seq_length in SEQ_LENGTHS:
        eps_typical = 0
        for _ in range(NUM_SIMS):
            xn = np.random.choice(NUM_SYMBOLS, seq_length, p=DIST)
            empirical_HX = np.sum(-np.log(DIST[xn]))/seq_length
            if np.abs(empirical_HX - HX) < EPSILON:
                eps_typical += 1
        p_eps_typical = eps_typical / NUM_SIMS
        eps_typicals.append(p_eps_typical)
        print(f'Sequence length: {seq_length}, Probability of being {EPSILON}-typical: {p_eps_typical}')

    plt.semilogx(SEQ_LENGTHS, eps_typicals, 'o-')
    plt.xlabel('Sequence length')
    plt.ylim([0, 1])
    plt.ylabel('Probability of being epsilon-typical')
    plt.title('Typicality of i.i.d. sequences')
    plt.grid()
    plt.show()

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    main()

# ------------------------------------------------------------------------------