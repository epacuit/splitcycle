'''
"Split Cycle: A New Condorcet Consistent Voting Method Independent of
Clones and Immune to Spoilers"
Wes Holliday and Eric Pacuit

---

The package, in its current form, is intended to allow easy use of the
SplitCycle method for determining Condorcet-consistent winners and
breaking Condorcet cycles that arise in ranked-choice elections.
Speed is prioritized over memory and resource consumption, though
currently no major optimizations have been implemented, since it is the
number of candidates, not the number of voters, that determines the time
it takes the SplitCycle algorithm, as implemented in this package, to
find all winners. With less than 100 candidates, the algorithm will find
the list of winners for most elections (even completely random ones)
almost instantly.

---

Adaptation as a Python package
Open source contributions by Ananth Venkatesh

<https://github.com/epacuit/splitcycle>
'''

__version__ = '1.0.0'

from . import core
from . import utils

# export user-facing functions
elect = core.elect
splitcycle = core.splitcycle

__all__ = ['elect', 'splitcycle', 'utils']
