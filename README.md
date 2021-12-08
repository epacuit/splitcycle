
# Stable Voting

README.md for the code used in the paper [Split Cycle: A New Condorcet Consistent Voting Method Independent of Clones and Immune to Spoilers](https://arxiv.org/abs/2004.02350) by Wes Holliday and Eric Pacuit.  

## Where to start

tl;dr: An overview of profiles (with ties) and voting methods is found in 01-Profiles.ipynb,  02-VotingMethods.ipynb, and 03-ProfileWithTies.ipynb.   See 04-SplitCycleExamples.ipynb for the implementation of Split Cycle and some examples from the paper. See 04-AnalyzingElectionData.ipynb for the code to anaylize actual elections from [preflib.org](https://preflib.org). 


1. 01-Profiles.ipynb Contains an overview of how to create profiles (implemented in voting/profiles.py) and generate profiles with different numbers of candidates/voters (implemented in voting/generate_profiles.py).   

A profile is created by initializing a Profile class object.  This needs a list of rankings (each ranking is a tuple of numbers), the number of candidates and a list giving the number of each ranking in the profile:

```python
from voting.profiles import Profile

rankings = [(0, 1, 2, 3), (2, 3, 1, 0), (3, 1, 2, 0), (1, 2, 0, 3), (1, 3, 2, 0)]
num_cands = 4
rcounts = [5, 3, 2, 4, 3]

prof = Profile(rankings, num_cands, rcounts=rcounts)
```

The function generate_profile is used to generate a profile for a given number of candidates and voters:  
```python
from voting.generate_profiles import generate_profile

# generate a profile using the Impartial Culture probability model
prof = generate_profile(3, 4) # prof is a Profile object with 3 candidate and 4 voters

# generate a profile using the Impartial Anonymous Culture probability model
prof = generate_profile(3, 4, probmod = "IAC") # prof is a Profile object with 3 candidate and 4 voters
```

2. Import and use voting methods (see voting/voting_methods.py for implementations and 02-VotingMethods.ipynb for an overview): 

```python
from voting.profiles import Profile
from voting.voting_methods import *

prof = Profile(rankings, num_cands, rcounts=rcounts)
print(f"The {borda.name} winners are {borda(prof)}")
```

3. For profiles in which voters submit strict weak orders (i.e., voters submit rankings with ties), see the class ProfileWithTies implemented in voting/profiles_with_ties.py

## Dev Notes

* Many of the voting methods assume  that voters submit linear orders over the set of candidates (as implemented in the Profile class). 
* In order to optimize some of the code for reasoning about profiles, it is assumed that in any profile the candidates are named by the initial segment of the non-negative integers.  So, in a profile with 5 candidates, the candidate names are "0, 1, 2, 3, and 4".   Use the `cmap` variable for different candidate names: `cmap` is a dictionary with keys 0, 1, ..., num_cands - 1 and values the "real" names of the candidates.  
* Not all voting methods work on ProfileWithTies.  In particular, borda and hare (ranked-choice) generates an error when given a ProfileWithTies

## Notebooks

1. 01-Profile.ipynb: This notebook is an overview of how to create profiles, remove candidates from a profile and generate profiles according to various probability models.    

2. 02-VotingMethods.ipynb: This notebook is an overview of the voting methods that are available. 

3. 03-ProfilesWithTies.ipynb: Profiles with voters that submit strict weak orders over the candidates. 

3. 04-SplitCycleExamples.ipynb: This notebook discusses the implementation of Split Cycle and a number of examples from the paper. 

4. 05-AnalyzingElectionData.ipynb: Analyzing actual elections from [preflib.org](https://preflib.org). 

## Other Files/Directories

1. voting/profiles.py: Implementation of the Profile class used to create and reason about profile (see 01-Profile.ipynb for an overview).

2. voting/profiles_with_ties.py: Implementation of the ProfileWithTies class used from elections in which voters submit strict weak orders (see 03-ProfilesWithTies.ipynb for an overview). 

3. voting/voting_methods.py: Implementations of the voting methods (see 02-VotingMethods.ipynb for an overview).

4. voting/generate_profiles.py: Implementation of  the function `generate_profile` to interface with the Preflib tools to generate profiles according to different probability models. 

5. preflib-data/: Data from [preflib.org](https://preflib.org) of actual elections discussed in the paper. 


## Requirements

All the code is written in Python 3. 

- [Preflib tools](https://github.com/PrefLib/PrefLib-Tools) (available in the voting/preflibtools directory)
- The notebooks and most of the library is built around a full SciPy stack: [MatPlotLib](https://matplotlib.org/), [Numpy](https://numpy.org/), [Pandas](https://pandas.pydata.org/)
- [numba](http://numba.pydata.org/) 
- [networkx](https://networkx.org/)
- [tabulate](https://github.com/astanin/python-tabulate)
- [seaborn](https://seaborn.pydata.org/)  
- [multiprocess](https://pypi.org/project/multiprocess/) (only needed if running the simulations in  05-ProbabilisticStabilityWinners.ipynb) 
- [tqdm.notebook](https://github.com/tqdm/tqdm)
 