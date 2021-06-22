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


def rank_ppr_ents(seed_list, sp_mat, mode="fixed", max_ent=500, min_ppr=0.005):
    seed = np.zeros((sp_mat.shape[0], 1))
    seed[seed_list] = 1. / len(set(seed_list))
    ppr = personalized_pagerank(seed, sp_mat, restart_prob=0.8, max_iter=20)
    if mode == "fixed":
        sorted_idx = np.argsort(ppr)[::-1]
        extracted_ents = sorted_idx[:max_ent]
        # check if any ppr values are nearly zero
        zero_idx = np.where(ppr[extracted_ents] < 1e-6)[0]
        if zero_idx.shape[0] > 0:
            extracted_ents = extracted_ents[:zero_idx[0]]
    else:
        extracted_ents = np.where(ppr > min_ppr)[0]
    return extracted_ents
