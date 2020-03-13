# Split Cycle

Split Cycle is a new Condorcet consistent voting method that is independent of clones and immune to spoilers.  Details about this  voting method, including the properties that it satisfies, can be found in the paper [Split Cycle: A New Condorcet Consistent Voting Method Independent of Clones and Immune to Spoilers]() by Wesley H. Holliday  and Eric Pacuit.

This repository contains implementations of a number of different voting methods, including Split Cycle and other Condorcet consistent methods, such as Beat Path, Ranked Pairs, Copeland, Minimax (also known as the Simpson-Kramer rule).  

## Contents of the repo

* voting/profile_optimized.py:  The ProfileOpt class is the main data structure to store and reason about a profile.   A **profile** for a set of candidates $$X$$ and voters $$V$$ is a function $$\mathbf{P}:V\rightarrow \mathcal{L}(X)$$, where $$\mathcal{L}(X)$$ is the set of linear orders of $$X$$.  
