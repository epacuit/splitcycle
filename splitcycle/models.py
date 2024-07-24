"""Voting model representations used in user-facing utilities"""

from random import randint
import numpy as np


def ic(n_ballots, n_candidates, ties):
    """
    Generate a random set of ballots according to the impartial culture
    model, with a parameter to specify whether there should be ties.

    `n_ballots`:
        the number of ballots in the election

    `n_candidates`:
        the number of candidates in the election

    `ties`:
        whether to allow ties in the election

    Return a numpy array of shape (n_ballots, n_candidates) that
    represents a preprocessed list of ballots with ranks 1 to
    `n_candidates` (can be used with `elect`)
    """
    if ties:
        # Generate random ballots with ties
        return np.random.randint(
            1, high=n_candidates+1, size=(n_ballots, n_candidates)
        )
    else:
        # Generate unique random ballots without ties
        ballots = np.zeros((n_ballots, n_candidates), dtype=int)
        for i in range(n_ballots):
            permutation = (
                np.random.permutation(n_candidates) + 1
            )  # Ensure 1 to n_candidates
            ballots[i] = permutation
        return ballots


def euclidean(n_ballots, n_candidates, n):
    """
    Generate a random set of ballots according to the euclidean
    (spatial) model in `n` dimensions.

    `n_ballots`:
        the number of ballots in the election

    `n_candidates`:
        the number of candidates in the election

    `n`:
        dimensionality of the voter preferences space

    Return a numpy array of shape (n_ballots, n_candidates) that
    represents a preprocessed list of ballots with ranks 1 to
    `n_candidates` (can be used with `elect`)
    """
    # generate random candidate points
    candidates = np.random.uniform(-1, 1, (n_candidates, n))

    # generate random voter points
    voters = np.random.uniform(-1, 1, (n_ballots, n))

    distances = np.linalg.norm(
        candidates[np.newaxis, :, :] - voters[:, np.newaxis, :], axis=2
    )

    return np.argsort(distances, axis=1) + 1
