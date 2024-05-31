import cython
import numpy as np
from Cython import nogil
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils import check_array


# cimport numpy as np
# cimport cython


def kmc2(X, k, chain_length=200, afkmc2=True, random_state=None, weights=None):
    """Cython implementation of k-MC2 and AFK-MC2 seeding

    Args:
      X: (n,d)-shaped np.ndarray with data points (or scipy CSR matrix)
      k: number of cluster centers
      chain_length: length of the MCMC chain
      afkmc2: Whether to run AFK-MC2 (if True) or vanilla K-MC2 (if False)
      random_state: numpy.random.RandomState instance or integer to be used as seed
      weights: n-sized np.ndarray with weights of data points (default: uniform weights)
    Returns:
      (k, d)-shaped numpy.ndarray with cluster centers
    """
    # Local cython variables
    # cdef np.intp_t j, curr_ind
    # cdef double cand_prob, curr_prob
    # cdef double[::1] q_cand, p_cand, rand_a

    # Handle input
    X = check_array(X, accept_sparse="csr", dtype=np.float64, order="C")
    sparse = not isinstance(X, np.ndarray)
    if weights is None:
        weights = np.ones(X.shape[0], dtype=np.float64)
    if random_state is None or isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)
    if not isinstance(random_state, np.random.RandomState):
        raise ValueError("RandomState should either be a numpy.random.RandomState"
                         " instance, None or an integer to be used as seed.")
    # Initialize result
    centers = np.zeros((k, X.shape[1]), np.float64, order="C")
    # Sample first center and compute proposal
    rel_row = X[random_state.choice(X.shape[0], p=weights / weights.sum()), :]
    centers[0, :] = rel_row.todense().flatten() if sparse else rel_row
    if afkmc2:
        di = np.min(euclidean_distances(X, centers[0:1, :], squared=True), axis=1) * weights
        q = di / np.sum(di) + weights / np.sum(weights)  # Only the potentials
    else:
        q = np.copy(weights)
    # Renormalize the proposal distribution
    q = q / np.sum(q)

    for i in range(k - 1):
        # Draw the candidate indices
        cand_ind = random_state.choice(X.shape[0], size=chain_length, p=q).astype(np.intp)
        # Extract the proposal probabilities
        q_cand = q[cand_ind]
        # Compute pairwise distances
        dist = euclidean_distances(X[cand_ind, :], centers[0:(i + 1), :], squared=True)
        # Compute potentials
        p_cand = np.min(dist, axis=1) * weights[cand_ind]
        # Compute acceptance probabilities
        rand_a = random_state.random_sample(size=chain_length)
        with nogil, cython.boundscheck(False), cython.wraparound(False), cython.cdivision(True):
            # Markov chain
            for j in range(q_cand.shape[0]):
                cand_prob = p_cand[j] / q_cand[j]
                if j == 0 or curr_prob == 0.0 or cand_prob / curr_prob > rand_a[j]:
                    # Init new chain             Metropolis-Hastings step
                    curr_ind = j
                    curr_prob = cand_prob
        rel_row = X[cand_ind[curr_ind], :]
        centers[i + 1, :] = rel_row.todense().flatten() if sparse else rel_row
    return centers
