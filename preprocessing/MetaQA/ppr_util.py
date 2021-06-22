import numpy as np


def personalized_pagerank(seed, W, restart_prob=0.8, max_iter=20):
    """Return the PPR vector for the given seed and restart prob.

    Args:
        seed: A sparse matrix of size E x 1.
        W: A sparse matrix of size E x E whose rows sum to one.
        restart_prob: A scalar in [0, 1].

    Returns:
        ppr: A vector of size E.
    """
    r = restart_prob * seed
    s = np.copy(r)
    for i in range(max_iter):
        r_new = (1. - restart_prob) * (W.transpose().dot(r))
        s = s + r_new
        delta = abs(r_new.sum())
        if delta < 1e-5:
            break
        r = r_new
    return np.squeeze(s)
