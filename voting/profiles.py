'''
    File: profile_optimized.py
    Author: Eric Pacuit (epacuit@umd.edu)
    Date: December 7, 2020
    
    Functions to reason about profiles
'''

from math import ceil
import numpy as np
from numba import jit
import networkx as nx
from tabulate import tabulate
import matplotlib.pyplot as plt

# turn off future warnings.
# getting the following warning when calling tabulate to display a profile: 
# /Library/Frameworks/Python.framework/Versions/3.8/lib/python3.8/site-packages/tabulate.py:1027: FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
#  if headers == "keys" and not rows:
# see https://stackoverflow.com/questions/40659212/futurewarning-elementwise-comparison-failed-returning-scalar-but-in-the-futur
#
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



# #######
# Internal compiled functions to optimize reasoning with profiles
# #######

@jit(fastmath=True)
def isin(arr, val):
    """optimized function testing if the value val is in the array arr
    """
    
    for i in range(arr.shape[0]):
        if (arr[i]==val):
            return True
    return False


@jit(nopython=True)
def _support(ranks, rcounts, c1, c2):
    """The number of voters that rank candidate c1 over candidate c2
    
    Parameters
    ----------
    rankings:  2d numpy array
        list of linear orderings of the candidates  
    rcounts:   1d numpy array
        list of number of voters for each ranking
    c1: int
        a candidate
    c2: int
        a candidate. 
    
    Key assumptions: 
        * the candidates are named 0...num_cands - 1, and c1 and c2 are 
          numbers between 0 and num_cands - 1
        * voters submit linear orders over the candidate        
    """
    
    diffs = ranks[0:,c1] - ranks[0:,c2] # for each voter, the difference of the ranks of c1 and c2
    diffs[diffs > 0] = 0 # c1 is ranked below c2  
    diffs[diffs < 0] = 1 # c1 is ranked above c2 
    num_rank_c1_over_c2 = np.multiply(diffs, rcounts) # mutliply by the number of each ranking
    return np.sum(num_rank_c1_over_c2)

@jit(fastmath=True,nopython=True)
def _margin(tally, c1, c2): 
    """The margin of c1 over c2: the number of voters that rank c1 over c2 minus 
    the number of voters that rank c2 over c1
    
    Parameters
    ----------
    tally:  2d numpy array
        the support for each pair of candidates  
    """
    return tally[c1][c2] - tally[c2][c1]


@jit(nopython=True)
def _num_rank(rankings, rcounts, cand, level):
    """The number of voters that rank cand at level 

    Parameters
    ----------
    rankings:  2d numpy array
        list of linear orderings of the candidates  
    rcounts:   1d numpy array
        list of number of voters for each ranking
    """
    cands_at_level =  rankings[0:,level-1] # get all the candidates at level
    is_cand = cands_at_level == cand # set to 0 each candidate not equal to cand
    return np.sum(is_cand * rcounts) 

@jit(nopython=True)
def _borda_score(rankings, rcounts, num_cands, cand):
    """The Borda score for cand 

    Parameters
    ----------
    rankings:  2d numpy array
        list of linear orderings of the candidates  
    rcounts:   1d numpy array
        list of number of voters for each ranking
    """
    
    bscores = np.arange(num_cands)[::-1]
    levels = np.arange(1,num_cands+1)
    num_ranks = np.array([_num_rank(rankings, rcounts, cand, level) for level in levels])
    return  np.sum(num_ranks * bscores)

@jit(nopython=True)
def _find_updated_profile(rankings, cands_to_ignore, num_cands):
    """Optimized method to remove all candidates from cands_to_ignore
    from a list of rankings. 
    
    Parameters
    ----------
    rankings:  2d numpy array
        list of linear orderings of the candidates  
    cands_to_ignore:  1d numpy array
        list of candidates to ignorme
    num_cands: int 
        the number of candidates in the original profile
    """
    updated_cand_num = num_cands - cands_to_ignore.shape[0]
    updated_prof_ranks = np.empty(shape=(rankings.shape[0],updated_cand_num), dtype=np.int32)
    
    for vidx in range(rankings.shape[0]):
        levels_idx = np.empty(num_cands - cands_to_ignore.shape[0], dtype=np.int32)
        _r = rankings[vidx]
        _r_level = 0
        for lidx in range(0, levels_idx.shape[0]): 
            for _r_idx in range(_r_level, len(_r)):
                if not isin(cands_to_ignore, _r[_r_idx]):
                    levels_idx[lidx]=_r_idx
                    _r_level = _r_idx + 1
                    break
        updated_prof_ranks[vidx] = np.array([_r[l] for l in levels_idx])
    return updated_prof_ranks

# #######
# Profiles
# #######

class Profile(object):
    """
    A profile of linear orderings

    In order to optimize the excecution of different voting methods, 
    there are two key assumptions: 
        1. Each voter has a linear ordering over the set of candidates
        2. The names of the candidates  are  0,..., num_cands - 1
    """
    def __init__(self, rankings, num_cands, rcounts=None, cmap=None):
        """
        Create a profile
                
        Parameters
        ----------
            rankings: 2d array
                each element is a linear ordering of the candidates
            num_cands: int
                number of candidates
            rcounts: 1d array
                optional parameter giving the total number of each ranking
            cmap: dict
                optional parameter mapping candidates to candidate names
        """
        
        self.num_cands = num_cands
        self.candidates = range(0, num_cands) 
        
        # linear ordering of the candidates for each voter
        self._rankings = np.array(rankings)   
        
        # for number of each ranking
        self._rcounts = np.array([1]*len(rankings)) if rcounts is None else np.array(rcounts) 
        
        # for each voter, the rankings of each candidate
        self._ranks = np.array([[_r.index(c) + 1 for c in self.candidates] for  _r in rankings])
        
        # 2d array where the c,d entery is the support of c over d
        self._tally = np.array([[_support(self._ranks, self._rcounts, c1, c2) 
                                 for c2 in self.candidates] for c1 in self.candidates ])
        
        # mapping candidates to candidate names
        self.cmap = cmap if cmap is not None else {c:c for c in self.candidates}
                
        # total number of voters
        self.num_voters = np.sum(self._rcounts)

    @property
    def rankings_counts(self):
        # getter function to get the rankings and rcounts
        return self._rankings, self._rcounts
    
    @property
    def rankings(self): 
        # get the list of rankings
        
        return [tuple(r) for ridx,r in enumerate(self._rankings) for n in range(self._rcounts[ridx])]
    
    def support(self, c1, c2):
        # the number of voters that rank c1 over c2 
        # wrapper function that calls the compiled _support function

        return self._tally[c1][c2]
    
    def margin(self, c1, c2):
        # the number of voters that rank c1 over c2 minus the number
        #   that rank c2 over c2.
        # wrapper function that calls the compiled _margin function

        return _margin(self._tally, c1, c2)
        
    def majority_prefers(self, c1, c2): 
        # return True if more voters rank c1 over c2 than c2 over c1

        return _margin(self._tally, c1, c2) > 0

    def num_rank(self, c, level): 
        # the number of voters that rank c at level 
        # wrapper that calls the compiled _num_rank function

        return _num_rank(self._rankings, self._rcounts, c, level=level)
        
    def plurality_scores(self):
        # return a dictionary of the plurality score for each candidate
        
        return {c: _num_rank(self._rankings, self._rcounts, c, level=1) for c in self.candidates}

    def borda_scores(self):
        # return a dictionary of the Borda scores for each candidate

        return {c: _borda_score(self._rankings, self._rcounts, self.num_cands, c) for c in self.candidates}

    def condorcet_winner(self):
        # return the Condorcet winner --- a candidate that is majority preferred to every other candidate
        # if a Condorcet winner doesn't exist, return None
        
        cw = None
        for c in self.candidates: 
            if all([self.majority_prefers(c,c2) for c2 in self.candidates if c != c2]): 
                cw = c
                break # if a Condorcet winner exists, then it is unique
        return cw

    def weak_condorcet_winner(self):
        # return the set of Weak Condorcet winner --- candidate c is a weak Condorcet winner if there 
        # is no other candidate majority preferred to c. Note that unlike with Condorcet winners, there 
        # may be more than one weak Condorcet winner.
        # if a weak Condorcet winner doesn't exist, return None
        
        weak_cw = list()
        for c in self.candidates: 
            if not any([self.majority_prefers(c2,c) for c2 in self.candidates if c != c2]): 
                weak_cw.append(c)
        return sorted(weak_cw) if len(weak_cw) > 0 else None

    def condorcet_loser(self):
        # return the Condorcet loser --- a candidate that is majority preferred by every other candidate
        # if a Condorcet loser doesn't exist, return None
        
        cl = None
        for c in self.candidates: 
            if all([self.majority_prefers(c2,c) for c2 in self.candidates if c != c2]): 
                cl = c
                break # if a Condorcet loser exists, then it is unique
        return cl
    
    def strict_maj_size(self):
        # return the size of  strictly more than 50% of the voters
        
        return int(self.num_voters/2 + 1 if self.num_voters % 2 == 0 else int(ceil(float(self.num_voters)/2)))

    def margin_graph(self, cmap=None): 
        # generate the margin graph (i.e., the weighted majority graph)
    
        mg = nx.DiGraph()
        mg.add_nodes_from(self.candidates)
        mg.add_weighted_edges_from([(c1, c2, self.margin(c1,c2))
                                    for c1 in self.candidates 
                                    for c2 in self.candidates if c1 != c2 if self.majority_prefers(c1, c2)])
        return mg

    def is_uniquely_weighted(self): 
        mg = self.margin_graph()
        has_zero_margins = any([not mg.has_edge(c,c2) and not mg.has_edge(c2,c) for c in mg.nodes for c2 in mg.nodes if c != c2])
        return not has_zero_margins and len(list(set([e[2] for e in mg.edges.data("weight") ]))) == len(mg.edges)

    def remove_candidates(self, cands_to_ignore):
        # remove all the candidates from cands_to_ignore from the profile
        # returns a new profile and a dictionary mapping new candidate names to the original names
        #   this is needed since we assume that candidates must be named 0...num_cands - 1
        
        updated_rankings = _find_updated_profile(self._rankings, np.array(cands_to_ignore), self.num_cands)
        new_num_cands = self.num_cands - len(cands_to_ignore)
        new_names = {c:cidx  for cidx, c in enumerate(sorted(updated_rankings[0]))}
        orig_names = {v:k  for k,v in new_names.items()}
        return Profile([[new_names[c] for c in r] for r in updated_rankings], new_num_cands, rcounts=self._rcounts, cmap=self.cmap), orig_names
    
    def display(self, cmap=None, style="pretty"):
        # display a profile
        # style defaults to "pretty" (the PrettyTable formatting)
        # other stype options is "latex" or "fancy_grid" (or any style option for tabulate)
        
        cmap = cmap if cmap is not None else self.cmap
        print(tabulate([[cmap[c] for c in cs] for cs in self._rankings.transpose()],
                       self._rcounts, tablefmt=style))        
        
    def display_margin_graph(self, cmap=None):
        # display the margin graph
        
        # create the margin graph.   The reason not to call the above method margin_graph 
        # is that we may want to apply the cmap to the names of the candidates
        
        cmap = cmap if cmap is not None else self.cmap
        
        mg = nx.DiGraph()
        mg.add_nodes_from([cmap[c] for c in self.candidates])
        mg.add_weighted_edges_from([(cmap[c1], cmap[c2], self.margin(c1,c2))
                                    for c1 in self.candidates 
                                    for c2 in self.candidates if c1 != c2 if self.majority_prefers(c1, c2)])

        pos = nx.circular_layout(mg)

        nx.draw(mg, pos, 
                font_size=20, font_color='white', node_size=700, 
                width=1.5, with_labels=True)
        labels = nx.get_edge_attributes(mg,'weight')
        nx.draw_networkx_edge_labels(mg,pos,edge_labels=labels, font_size=14, label_pos=0.3)
        plt.show()

    def __str__(self): 
        # print the profile as a table
        
        return tabulate([[self.cmap[c] for c in cs] for cs in self._rankings.transpose()],
                        self._rcounts, tablefmt="pretty")    

def simple_lift(ranking, c):
    assert c != ranking[0], "can't lift a candidate already in first place"
    
    c_idx = ranking.index(c)
    ranking[c_idx - 1], ranking[c_idx] = ranking[c_idx],ranking[c_idx-1]
    return ranking

def simple_drop(ranking, c):
    assert c != ranking[-1], "can't drop a candidate already in last place"
    
    c_idx = ranking.index(c)
    ranking[c_idx + 1], ranking[c_idx] = ranking[c_idx],ranking[c_idx + 1]
    return ranking

def mg_to_wmg(mg):
    
    wmg = nx.DiGraph()
    wmg.add_nodes_from(mg.nodes)
    wmg.add_edges_from(mg.edges(data=True))
    
    for c in mg.nodes: 
        for c2 in mg.nodes:
            if c != c2 and not mg.has_edge(c,c2) and not mg.has_edge(c2,c):
                wmg.add_edge(c,c2)
                wmg[c][c2]['weight'] = 0
                wmg.add_edge(c2,c)
                wmg[c2][c]['weight'] = 0
    return wmg

def main():
    
    print("Create a profile")
    print("Profile([(0,1,2), (1,2,0), (2,0,1)], 3)")
    prof = Profile([(0,1,2), (1,2,0), (2,0,1)], 3)
    print("Display the profile using either prof.dipaly() or print(prof)")
    prof.display()
    print("A number of operations on profiles")
    c1 = 1
    c2 = 2
    print(f"support of {c1} over {c2} is ", prof.support(c1, c2))
    print(f"margin of {c1} over {c2} is ", prof.margin(c1, c2))
    print(f"{c1} is  majority preferred to {c2}:  ", prof.majority_prefers(c1, c2))
    print(f"The plurality scores:  ", prof.plurality_scores())
    print(f"The Borda scores:  ", prof.borda_scores())
    print(f"The Condorcet winner (should be none since there is no Condorcet winner in the profile):  {prof.condorcet_winner()}")
    print(f"The Condorcet loser (should be none since there is no Condorcet loser in the profile): {prof.condorcet_loser()}")


if __name__ == "main":
    main()
