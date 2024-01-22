'''
User-facing utilities for data processing and interacting with the
SplitCycle package
'''

from random import randint
from tabulate import tabulate
import numpy as np
from .core import margins_from_ballots
from .errors import not_enough_candidates


def augment(ballots):
    '''
    Given a list of `ballots` (as described in `elect`, but with all
    ranked candidates having positive rank, replace all unranked
    candidates (represented with negative rank) with a rank greater than
    that of the least preferred ranked candidate in that ballot.
    '''
    for i, ballot in enumerate(ballots):
        last = -1  # keep track of last unranked candidate if needed
        for j, rank in enumerate(ballot):
            if rank < 0:
                if last == -1:
                    # store last unranked candidate if not already set
                    last = max(ballot) + 1

                # replace rank
                ballots[i][j] = last

    return ballots


def info(ballots, candidates, verbose=True):
    '''
    Given preprocessed `ballots` and `candidates` objects (as described
    in `elect`), return a dictionary of information about the election
    to ensure all data was processed correctly. Turn off print output
    with `verbose=False`.
    '''
    if ballots.shape[1] != len(candidates):
        not_enough_candidates()

    n_candidates = len(candidates)
    n_ballots = ballots.shape[0]
    margins = margins_from_ballots(ballots)
    ex_ballot = ballots[randint(0, n_ballots - 1)]

    if verbose:
        print(f'There are {n_candidates} candidates in this election.')
        print(f'There are {n_ballots} ballots ranking these candidates:')
        print(', '.join(candidates))
        print('The associated margins graph for this election is below:')
        print('Left candidate vs top candidate')
        margins_str = [[str(cell) for cell in row] for row in margins]
        labeled_margins = [
            [candidates[i]] + row for i, row in enumerate(margins_str)
        ]
        row_labels = [''] + candidates
        print(tabulate(labeled_margins, headers=row_labels, tablefmt='pretty'))
        print('Here is an example ballot:')
        # rank candidate names according to ranks in ballot
        ranked_candidates = [
            candidate for _, candidate in sorted(zip(ex_ballot, candidates))
        ]
        print(', '.join(ranked_candidates))
        print('Candidate: Rank (lower is better)')
        print('---')
        for i, candidate in enumerate(candidates):
            print(f'{candidate}: {ex_ballot[i]}')

    return {
        'n_candidates': n_candidates,
        'n_ballots': n_ballots,
        'margins': margins,
        'ex_ballot': ex_ballot,
    }


def gen_random_ballots(n_ballots, n_candidates, ties=True):
    '''
    Generate a random set of ballots for testing purposes*

    *Note that this does NOT accurately represent voting in real
    elections, and should only be used for testing modules in the
    SplitCycle package

    `n_ballots`:
        the number of ballots in the election
    
    `n_candidates`:
        the number of candidates in the election
    
    `ties=True`:
        whether or not to allow ties between candidates (if ties are not
        allowed, this may slow down generation)

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


def expected_utility(p, r, c, y, t):
    '''See (Masciandaro 2007)'''
    return (1 - p) * (1 + r - c) * y - p * (t * y ** 2 + c * y)


def optimize(p, r, c, t):
    '''See (Masciandaro 2007)'''
    return ((1 + r) * (1 - p) - c) / (2 * p * t)
