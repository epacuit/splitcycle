<!-- #region -->

# Split Cycle

README.md for the code used in the paper [Split Cycle: A New Condorcet Consistent Voting Method Independent of Clones and Immune to Spoilers](https://arxiv.org/abs/2004.02350) by Wes Holliday and Eric Pacuit.  


The notebooks use the Python package ``pref_voting``.   Consult [https://pref-voting.readthedocs.io/](https://pref-voting.readthedocs.io/) for an overview of this package.  

## Notebooks

* 01-SplitCycle.ipynb: An efficient algorithm for computing Split Cycle.
* 02-SplitCycleExamples.ipynb: All the examples from the paper. 
* 03-AppendixCExamples.ipynb: All the examples from Appendix C. 
* 04-AppendixDIrresolutenessData.ipynb: The code to produce the graphs in Appendix D. 
* 05-Tables.ipynb: The code to generate the three tables in the paper. 


## Requirements

All the code is written in Python 3. The notebooks use the following libraries: 

- [Preferential Voting Tools](https://pref-voting.readthedocs.io/en/latest/)
- The notebooks and the pref_voting library is built around a full SciPy stack: [MatPlotLib](https://matplotlib.org/), [Numpy](https://numpy.org/), [Pandas](https://pandas.pydata.org/), [numba](http://numba.pydata.org/), [networkx](https://networkx.org/), and [tabulate](https://github.com/astanin/python-tabulate)
- [tqdm.notebook](https://github.com/tqdm/tqdm)

