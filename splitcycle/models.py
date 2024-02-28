'''Voting model representations used in user-facing utilities'''

from random import randint
import numpy as np


def ic(n_ballots, n_candidates, ties):
    '''
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
    '''
    # generate random ballots
    ballots = np.zeros((n_ballots, n_candidates))
    for i in range(n_ballots):
        if ties:
            ballots[i] = np.array(
                [randint(1, n_candidates) for _ in range(n_candidates)]
            )
        else:
            # generate `n_candidates` unique random integers
            # from 1 to `n_candidates`
            ballots[i] = 1 + np.random.choice(
                n_candidates, n_candidates, replace=False
            )

    return ballots


def euclidean(n_ballots, n_candidates, n):
    '''
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
    '''
    # generate random candidate points
    candidates = np.random.uniform(-1, 1, (n_candidates, n))

    # generate random voter points
    voters = np.random.uniform(-1, 1, (n_ballots, n))

    # compute voter preferences
    ballots = np.zeros((n_ballots, n_candidates))
    for i in range(n_ballots):
        distances = np.linalg.norm(candidates - voters[i], axis=1)
        ballots[i] = 1 + np.argsort(distances)

    return ballots
