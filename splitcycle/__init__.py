'''
"Split Cycle: A New Condorcet Consistent Voting Method Independent of Clones and Immune to Spoilers"
Wes Holliday and Eric Pacuit

Adaptation as a Python package
Open source contributions by Ananth Venkatesh and Youwen Wu
'''

__version__ = '1.0.0'

import numpy as np

from .core import *

def elect(margins, candidates=None):
    splitcycle(margins, candidates)
