'''
"Split Cycle: A New Condorcet Consistent Voting Method Independent of
Clones and Immune to Spoilers"
Wes Holliday and Eric Pacuit

Adaptation as a Python package
Open source contributions by Ananth Venkatesh and Youwen Wu
'''

__version__ = '1.0.0'

from . import core


def elect(margins, candidates=None):
    '''Alias for `splitcycle`'''
    core.splitcycle(margins, candidates)
