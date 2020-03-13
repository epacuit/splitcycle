# Split Cycle

Split Cycle is a new Condorcet consistent voting method that is independent of clones and immune to spoilers.  Details about this  voting method, including the properties that it satisfies, can be found in the paper [Split Cycle: A New Condorcet Consistent Voting Method Independent of Clones and Immune to Spoilers]() by Wesley H. Holliday  and Eric Pacuit.

This repository contains implementations of a number of different voting methods, including Split Cycle and other Condorcet consistent methods, such as Beat Path, Ranked Pairs, Copeland, Minimax (also known as the Simpson-Kramer rule).  

## Contents of the Repo

* voting/profile_optimized.py:  The ProfileOpt class is the main data structure to store and reason about a profile.   A **profile** for a set of candidates  and voters is a function that assigns to each voter a linear order on the set of candidates.  The easiest way to create a profile:

```python
# use integers for the candidate names
cand_names = [0, 1, 2]

# dictionary mapping candidate names to strings (to make profiles easier to read)
cmap = {0:"a", 1:"b", 2:"c"}

# an anonymous profile is a dictionary with linear orders over cand_names as keys each associated 
# with the number of voters having that ranking.
anon_prof = {(1, 0, 2): 2,
             (0, 2, 1): 3,
             (2, 1, 0): 4} 

# create a profile given the anonymous profile,  the candidate names and candidate map
prof = create_profile_opt_from_anon_profile(anon_prof, candidate_names = cand_names, cmap = cmap) 
```
The implementation of the profile operations is optimized using jit from [numba](https://numba.pydata.org/)

* voting/voting_methods_for_optimized.py: Implementation of a number of different voting methods.  Each voting method takes a ProfileOpt as a parameter and returns a sorted list of candidates (a sorted list of cand_names).  The voting methods that are implemented include: plurality, majority, borda, plurality_with_runoff, hare (also known as "Ranked Choice"), coombs, strict_nanson (and weak_nanson), baldwin, condorcet, schulze_beatpath, and ranked_pairs.   (See the paper and the [Voting Methods](https://plato.stanford.edu/entries/voting-methods/) entry for definitions of these voting methods).  The implemenation  of Split Cycle (split_cycle) is found in SplitCycleExamples.ipynb.  Faster versions of Split Cycle and Beat Path (called splitcycle_faster and beatpath_faster) is found in SplitCycleExamples.ipynb.  

* SplitCycleExamples.ipynb: A notebook with all the examples discussed in the paper [Split Cycle: A New Condorcet Consistent Voting Method Independent of Clones and Immune to Spoilers](). 

* preflibtools/: These are the tools from [preflib.org](http://www.preflib.org/).   The code can be found here: [https://github.com/PrefLib/PrefLib-Tools](https://github.com/PrefLib/PrefLib-Tools).  This code is only needed if generating profiles (e.g., using the impartial culture or mallows model).  In particular, it is not needed to run the SplitCycleExamples.ipynb notebook.  

* generate_profiles.py: Helper functions to interface between the ProfileOpt data structure and the outputs from the preflibtools generate profile functions.

## Requirements

* [matplotlib.pyplot](https://matplotlib.org/) (to display margin graphs)
* [networkx](https://networkx.github.io/) (to store and reason about margin graphs)
* [numba.jit](https://numba.pydata.org/) (to speed up some calculations involving profiles)
* [numpy](https://numpy.org/) (for average and to opitmize storing profiles)
* itertools (combinations, product and permutations)
* math.ceil (for some calculations)
* random (to implement a random tiebreaking rule)
* string (to generate voter/candidate names)
