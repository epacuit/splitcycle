'''
User-facing utilities for data processing and interacting with the
SplitCycle package
'''


def format_ballots(ballots, candidates):
    '''
    Given a list of `ballots` (as described in `elect`, but with
    candidate names instead of candidate indices) and a list of
    candidate names `candidates` (described in `elect`), replace each
    ballot with a list of candidate indices so it can be used with the
    SplitCycle package
    '''
    for i, ballot in enumerate(ballots):
        for j, rank in enumerate(ballot):
            for k, name in enumerate(rank):
                ballots[i][j][k] = candidates.index(name)

    return ballots
