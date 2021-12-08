'''
    File: voting_methods.py
    Author: Eric Pacuit (epacuit@umd.edu)
    Date: November 6, 2021
    
    Implementations of voting methods
'''

from .profiles import  _borda_score, _find_updated_profile
import matplotlib.pyplot as plt
from itertools import permutations, product
import networkx as nx
import numpy as np
import math
from numba import jit
import math
'''TODO: 
   * to optimize the iterative methods, I am currently using the private compiled methods
     _borda_score and _find_updated_profiles. We should think of a better way to deal with this issue. 
   * implement other voting methods: e.g., Dodgson, Young, Kemeny
   * implement the linear programming version of Ranked Pairs: https://arxiv.org/pdf/1805.06992.pdf
   * implement a Voting Method class?
'''

# #####
# Helper functions
# #####

# decorator that adds a "name" attribute to a voting method function
def vm_name(vm_name):
    def wrapper(f):
        f.name = vm_name
        return f
    return wrapper

@jit(fastmath=True)
def isin(arr, val):
    """compiled function testing if the value val is in the array arr
    """
    
    for i in range(arr.shape[0]):
        if (arr[i]==val):
            return True
    return False

@jit(nopython=True)
def _num_rank_first(rankings, rcounts, cands_to_ignore, cand):
    """The number of voters that rank candidate cand first after ignoring the candidates in 
    cands_to_ignore
    
    Parameters
    ----------
    rankings:  2d numpy array
        list of linear orderings of the candidates  
    rcounts:  1d numpy array
        list of numbers of voters with the rankings  
    cands_to_ignore:   1d numpy array
        list of the candidates to ignore
    cand: int
        a candidate
    
    Key assumptions: 
        * the candidates are named 0...num_cands - 1, and c1 and c2 are 
          numbers between 0 and num_cands - 1
        * voters submit linear orders over the candidate        
    """
    
    num_voters = len(rankings)
    
    top_cands_indices = np.zeros(num_voters, dtype=np.int32)
    
    for vidx in range(num_voters): 
        for level in range(0, len(rankings[vidx])):
            if not isin(cands_to_ignore, rankings[vidx][level]):
                top_cands_indices[vidx] = level
                break                
    top_cands = np.array([rankings[vidx][top_cands_indices[vidx]] for vidx in range(num_voters)])
    is_cand = top_cands == cand # set to 0 each candidate not equal to cand
    return np.sum(is_cand * rcounts) 


@jit(nopython=True)
def _num_rank_last(rankings, rcounts, cands_to_ignore, cand):
    """The number of voters that rank candidate cand last after ignoring the candidates in 
    cands_to_ignore
    
    Parameters
    ----------
    rankings:  2d numpy array
        list of linear orderings of the candidates  
    rcounts:  1d numpy array
        list of numbers of voters with the rankings  
    cands_to_ignore:   1d numpy array
        list of the candidates to ignore
    cand: int
        a candidate
    
    Key assumptions: 
        * the candidates are named 0...num_cands - 1, and c1 and c2 are 
          numbers between 0 and num_cands - 1
        * voters submit linear orders over the candidate        
    """
    
    num_voters = len(rankings)
    
    last_cands_indices = np.zeros(num_voters, dtype=np.int32)
    
    for vidx in range(num_voters): 
        for level in range(len(rankings[vidx]) - 1,-1,-1):
            if not isin(cands_to_ignore, rankings[vidx][level]):
                last_cands_indices[vidx] = level
                break                
    bottom_cands = np.array([rankings[vidx][last_cands_indices[vidx]] for vidx in range(num_voters)])
    is_cand = bottom_cands  == cand
    return np.sum(is_cand * rcounts) 

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

@vm_name("Majority")
def majority(profile):
    '''returns the majority winner is the candidate with a strict majority  of first place votes.   
    Returns an empty list if there is no candidate with a strict majority of first place votes.
    '''
    
    maj_size = profile.strict_maj_size()
    plurality_scores = profile.plurality_scores()
    maj_winner = [c for c in profile.candidates if  plurality_scores[c] >= maj_size]
    return sorted(maj_winner)


# #####
# Scoring rules
# ####

@vm_name("Plurality")
def plurality(profile):
    """A plurality winnner a candidate that is ranked first by the most voters
    """

    plurality_scores = profile.plurality_scores()
    max_plurality_score = max(plurality_scores.values())
    
    return sorted([c for c in profile.candidates if plurality_scores[c] == max_plurality_score])

@vm_name("Borda")
def borda(profile):
    """A Borda winner is a candidate with the larget Borda score. 
    
    The Borda score of the candidates is calculated as follows: If there are $m$ candidates, then 
    the Borda score of candidate $c$ is \sum_{r=1}^{m (m - r) * Rank(c,r)$ where $Rank(c,r)$ is the 
    number of voters that rank candidate $c$ in position $r$. 
    """
    
    candidates = profile.candidates
    borda_scores = profile.borda_scores()
    max_borda_score = max(borda_scores.values())
    
    return sorted([c for c in candidates if borda_scores[c] == max_borda_score])

@vm_name("Anti-Plurality")
def anti_plurality(profile):
    """An anti-plurality winnner is a candidate that is ranked last by the fewest voters"""
    
    candidates, num_candidates = profile.candidates, profile.num_cands
    last_place_scores = {c: profile.num_rank(c,level=num_candidates) for c in candidates}
    min_last_place_score = min(list(last_place_scores.values()))
    
    return sorted([c for c in candidates if last_place_scores[c] == min_last_place_score])

# #####
# Iterative Methods
# ####

@vm_name("Instant Runoff")
def hare(profile):
    """If there is a majority winner then that candidate is the ranked choice winner
    If there is no majority winner, then remove all candidates that are ranked first by the fewest 
    number of voters.  Continue removing candidates with the fewest number first-place votes until 
    there is a candidate with a majority of first place votes.  
    
    Note: If there is  more than one candidate with the fewest number of first-place votes, then *all*
    such candidates are removed from the profile. 
    
    Note: This is known as "Ranked Choice", "Hare", "IRV", "Alternative Voter" or "STV" 
    """
    
    num_cands = profile.num_cands   
    candidates = profile.candidates
    strict_maj_size = profile.strict_maj_size()
    
    rs, rcounts = profile.rankings_counts # get all the ranking data
    
    cands_to_ignore = np.empty(0)

    winners = [c for c in candidates 
               if _num_rank_first(rs, rcounts, cands_to_ignore, c) >= strict_maj_size]

    while len(winners) == 0:
        plurality_scores = {c: _num_rank_first(rs, rcounts, cands_to_ignore, c) for c in candidates 
                            if  not isin(cands_to_ignore,c)}  
        min_plurality_score = min(plurality_scores.values())
        lowest_first_place_votes = np.array([c for c in plurality_scores.keys() 
                                             if  plurality_scores[c] == min_plurality_score])

        # remove cands with lowest plurality winners
        cands_to_ignore = np.concatenate((cands_to_ignore, lowest_first_place_votes), axis=None)
        if len(cands_to_ignore) == num_cands: #all the candidates where removed
            winners = sorted(lowest_first_place_votes)
        else:
            winners = [c for c in candidates 
                       if not isin(cands_to_ignore,c) and _num_rank_first(rs, rcounts, cands_to_ignore, c) >= strict_maj_size]
     
    return sorted(winners)

@vm_name("Instant Runoff TB")
def hare_tb(profile, tie_breaker = None):
    """If there is a majority winner then that candidate is the ranked choice winner
    If there is no majority winner, then remove all candidates that are ranked first by the fewest 
    number of voters.  Continue removing candidates with the fewest number first-place votes until 
    there is a candidate with a majority of first place votes.  
    
    Note: If there is  more than one candidate with the fewest number of first-place votes, then *all*
    such candidates are removed from the profile. 
    
    Note: This is known as "Ranked Choice", "Hare", "IRV", "Alternative Voter" or "STV" 
    """
    # the tie_breaker is any linear order (i.e., list) of the candidates
    tb = tie_breaker if tie_breaker is not None else list(range(profile.num_cands))
    
    num_cands = profile.num_cands   
    candidates = profile.candidates
    strict_maj_size = profile.strict_maj_size()
    
    rs, rcounts = profile.rankings_counts # get all the ranking data
    
    cands_to_ignore = np.empty(0)

    winners = [c for c in candidates 
               if _num_rank_first(rs, rcounts, cands_to_ignore, c) >= strict_maj_size]

    while len(winners) == 0:
        plurality_scores = {c: _num_rank_first(rs, rcounts, cands_to_ignore, c) for c in candidates 
                            if  not isin(cands_to_ignore,c)}  
        min_plurality_score = min(plurality_scores.values())
        lowest_first_place_votes = np.array([c for c in plurality_scores.keys() 
                                             if  plurality_scores[c] == min_plurality_score])
        
        cand_to_remove = lowest_first_place_votes[0]
        for c in lowest_first_place_votes[1:]: 
            if tb.index(c) < tb.index(cand_to_remove):
                cand_to_remove = c

        # remove cands with lowest plurality winners
        cands_to_ignore = np.concatenate((cands_to_ignore, cand_to_remove), axis=None)
        if len(cands_to_ignore) == num_cands: #all the candidates where removed
            winners = sorted(lowest_first_place_votes)
        else:
            winners = [c for c in candidates 
                       if not isin(cands_to_ignore,c) and _num_rank_first(rs, rcounts, cands_to_ignore, c) >= strict_maj_size]
     
    return sorted(winners)

@vm_name("Instant Runoff PUT")
def hare_put(profile):
    """Instant Runoff with parallel universe tie-breaking (PUT).  Apply the Instant Runoff method with a tie-breaker
    for each possible linear order over the candidates. 
    
    Warning: This will take a long time on profiles with many candidates."""
    
    candidates = profile.candidates
    strict_maj_size = profile.strict_maj_size()
    
    rs, rcounts = profile.rankings_counts # get all the ranking data
    
    winners = [c for c in candidates 
               if _num_rank_first(rs, rcounts, np.empty(0), c) >= strict_maj_size]

    if len(winners) == 0:
        # run Instant Runoff with tie-breaker for each permulation of candidates
        for tb in permutations(candidates):
            winners += hare_tb(profile, tie_breaker = tb) 

    return sorted(list(set(winners)))

@vm_name("Instant Runoff")
def instant_runoff(profile): 
    return hare(profile)

@vm_name("Instant Runoff")
def instant_runoff_put(profile): 
    return hare_put(profile)

@vm_name("PluralityWRunoff")
def plurality_with_runoff(profile):
    """If there is a majority winner then that candidate is the plurality with runoff winner
    If there is no majority winner, then hold a runoff with  the top two candidates: 
    either two (or more candidates)  with the most first place votes or the candidate with 
    the most first place votes and the candidate with the 2nd highest first place votes 
    are ranked first by the fewest number of voters.  
    
    A candidate is a Plurality with Runoff winner if it is a winner in a runoff between two pairs of 
    first- or second- ranked candidates. 
    
    Note: If the candidates are all tied for the most first place votes, then all candidates are winners. 
    """
    
    candidates = profile.candidates
    rs, rcounts = profile.rankings_counts # get all the ranking data
    
    cands_to_ignore = np.empty(0)
    plurality_scores = profile.plurality_scores()  

    max_plurality_score = max(plurality_scores.values())
    
    first = [c for c in candidates if plurality_scores[c] == max_plurality_score]
    second = list()
    if len(first) == 1:
        second_plurality_score = list(reversed(sorted(plurality_scores.values())))[1]
        second = [c for c in candidates if plurality_scores[c] == second_plurality_score]
       
    
    #print("runoff ", runoff_candidates)
    
    if len(second) > 0:
        all_runoff_pairs = product(first, second)
    else: 
        all_runoff_pairs = [(c1,c2) for c1,c2 in product(first, first) if c1 != c2]

    winners = list()
    for c1, c2 in all_runoff_pairs: 
        
        if profile.margin(c1,c2) > 0:
            winners.append(c1)
        elif profile.margin(c1,c2) < 0:
            winners.append(c2)
        elif profile.margin(c1,c2) == 0:
            winners.append(c1)
            winners.append(c2)
    
    return sorted(list(set(winners)))
####
# Removed following version of Plurality with Runoff that deals with ties differently than the one above
#
# @vm_name("PluralityWRunoff")
# def plurality_with_runoff(profile):
#     """If there is a majority winner then that candidate is the plurality with runoff winner
#     If there is no majority winner, then hold a runoff with  the top two candidates: 
#     either two (or more candidates)  with the most first place votes or the candidate with 
#     the most first place votes and the candidate with the 2nd highest first place votes 
#     are ranked first by the fewest number of voters.    

#     Note: If the candidates are all tied for the most first place votes, then all candidates are winners. 
#     """

#     candidates = profile.candidates
#     rs, rcounts = profile.rankings_counts # get all the ranking data

#     cands_to_ignore = np.empty(0)
#     plurality_scores = profile.plurality_scores()  

#     max_plurality_score = max(plurality_scores.values())

#     first = [c for c in candidates if plurality_scores[c] == max_plurality_score]

#     if len(first) > 1:
#         runoff_candidates = first
#     else:
#         # find the 2nd highest plurality score
#         second_plurality_score = list(reversed(sorted(plurality_scores.values())))[1]
#         second = [c for c in candidates if plurality_scores[c] == second_plurality_score]
#         runoff_candidates = first + second

#     runoff_candidates = np.array(runoff_candidates)
#     candidates_to_ignore = np.array([c for c in candidates if not isin(runoff_candidates,c)])

#     runoff_plurality_scores = {c: _num_rank_first(rs, rcounts, candidates_to_ignore, c) for c in candidates 
#                                if isin(runoff_candidates,c)} 

#     runoff_max_plurality_score = max(runoff_plurality_scores.values())

#     return sorted([c for c in runoff_plurality_scores.keys() 
#                    if runoff_plurality_scores[c] == runoff_max_plurality_score])

@vm_name("Coombs")
def coombs(profile):
    """If there is a majority winner then that candidate is the Coombs winner
    If there is no majority winner, then remove all candidates that are ranked last by the greatest 
    number of voters.  Continue removing candidates with the most last-place votes until 
    there is a candidate with a majority of first place votes.  
    
    Note: If there is  more than one candidate with the most number of last-place votes, then *all*
    such candidates are removed from the profile. 
    """
    
    num_cands = profile.num_cands   
    candidates = profile.candidates
    strict_maj_size = profile.strict_maj_size()
    
    rs, rcounts = profile.rankings_counts # get all the ranking data
    
    cands_to_ignore = np.empty(0)

    winners = [c for c in candidates 
               if _num_rank_first(rs, rcounts, cands_to_ignore, c) >= strict_maj_size]

    while len(winners) == 0:
        
        last_place_scores = {c: _num_rank_last(rs, rcounts, cands_to_ignore, c) for c in candidates 
                             if not isin(cands_to_ignore,c)}  
        max_last_place_score = max(last_place_scores.values())
        greatest_last_place_votes = np.array([c for c in last_place_scores.keys() 
                                              if  last_place_scores[c] == max_last_place_score])

        # remove candidates ranked last by the greatest number of voters
        cands_to_ignore = np.concatenate((cands_to_ignore, greatest_last_place_votes), axis=None)
        
        if len(cands_to_ignore) == num_cands:
            winners = list(greatest_last_place_votes)
        else:
            winners = [c for c in candidates 
                       if not isin(cands_to_ignore,c) and _num_rank_first(rs, rcounts, cands_to_ignore, c) >= strict_maj_size]

    return sorted(winners)

@vm_name("Coombs")
def coombs_with_data(profile):
    """If there is a majority winner then that candidate is the Coombs winner
    If there is no majority winner, then remove all candidates that are ranked last by the greatest 
    number of voters.  Continue removing candidates with the most last-place votes until 
    there is a candidate with a majority of first place votes.  
    
    Note: If there is  more than one candidate with the most number of last-place votes, then *all*
    such candidates are removed from the profile. 
    
    Returns the order of elimination
    """
    
    num_cands = profile.num_cands   
    candidates = profile.candidates
    strict_maj_size = profile.strict_maj_size()
    
    rs, rcounts = profile.rankings_counts # get all the ranking data
    
    cands_to_ignore = np.empty(0)

    winners = [c for c in candidates 
               if _num_rank_first(rs, rcounts, cands_to_ignore, c) >= strict_maj_size]

    elims_list = list()
    while len(winners) == 0:
        
        last_place_scores = {c: _num_rank_last(rs, rcounts, cands_to_ignore, c) for c in candidates 
                             if not isin(cands_to_ignore,c)}  
        max_last_place_score = max(last_place_scores.values())
        greatest_last_place_votes = np.array([c for c in last_place_scores.keys() 
                                              if  last_place_scores[c] == max_last_place_score])

        elims_list.append(list(greatest_last_place_votes))
        # remove candidates ranked last by the greatest number of voters
        cands_to_ignore = np.concatenate((cands_to_ignore, greatest_last_place_votes), axis=None)
        
        if len(cands_to_ignore) == num_cands:
            winners = list(greatest_last_place_votes)
        else:
            winners = [c for c in candidates 
                       if not isin(cands_to_ignore,c) and _num_rank_first(rs, rcounts, cands_to_ignore, c) >= strict_maj_size]

    return sorted(winners), elims_list
###
# Variations of Coombs with tie-breaking, parallel universe tie-breaking
##

@vm_name("Coombs TB")
def coombs_tb(profile, tie_breaker=None):
    """Coombs with a fixed tie-breaking rule:  If there is a majority winner then that candidate 
    is the Coombs winner.  If there is no majority winner, then remove all candidates that 
    are ranked last by the greatest  number of voters.   If there are ties, then choose the candidate
    according to a fixed tie-breaking rule (given below). Continue removing candidates with the 
    most last-place votes until     there is a candidate with a majority of first place votes.  
    
    The tie-breaking rule is any linear order (i.e., list) of the candidates.  The default rule 
    is to order the candidates as follows: 0,....,num_cands-1

    """
    
    # the tie_breaker is any linear order (i.e., list) of the candidates
    tb = tie_breaker if tie_breaker is not None else list(range(profile.num_cands))

    num_cands = profile.num_cands   
    candidates = profile.candidates
    strict_maj_size = profile.strict_maj_size()
    
    rs, rcounts = profile.rankings_counts # get all the ranking data
    
    cands_to_ignore = np.empty(0)

    winners = [c for c in candidates 
               if _num_rank_first(rs, rcounts, cands_to_ignore, c) >= strict_maj_size]

    while len(winners) == 0:
        
        last_place_scores = {c: _num_rank_last(rs, rcounts, cands_to_ignore, c) for c in candidates 
                             if not isin(cands_to_ignore,c)}  
        max_last_place_score = max(last_place_scores.values())
        greatest_last_place_votes = [c for c in last_place_scores.keys() 
                                     if  last_place_scores[c] == max_last_place_score]

        # select the candidate to remove using the tie-breaking rule (a linear order over the candidates)
        cand_to_remove = greatest_last_place_votes[0]
        for c in greatest_last_place_votes[1:]: 
            if tb.index(c) < tb.index(cand_to_remove):
                cand_to_remove = c
        
        # remove candidates ranked last by the greatest number of voters
        cands_to_ignore = np.concatenate((cands_to_ignore, np.array([cand_to_remove])), axis=None)
        
        if len(cands_to_ignore) == num_cands:
            winners = list(greatest_last_place_votes)
        else:
            winners = [c for c in candidates 
                       if not isin(cands_to_ignore,c) and _num_rank_first(rs, rcounts, cands_to_ignore, c) >= strict_maj_size]

    return sorted(winners)

@vm_name("Coombs PUT")
def coombs_put(profile):
    """Coombs with parallel universe tie-breaking (PUT).  Apply the Coombs method with a tie-breaker
    for each possible linear order over the candidates. 
    
    Warning: This will take a long time on profiles with many candidates."""
    
    candidates = profile.candidates
    strict_maj_size = profile.strict_maj_size()
    
    rs, rcounts = profile.rankings_counts # get all the ranking data
    
    winners = [c for c in candidates 
               if _num_rank_first(rs, rcounts, np.empty(0), c) >= strict_maj_size]

    if len(winners) == 0:
        # run Coombs with tie-breaker for each permulation of candidates
        for tb in permutations(candidates):
            winners += coombs_tb(profile, tie_breaker = tb) 

    return sorted(list(set(winners)))

# ##

@vm_name("Baldwin")
def baldwin(profile):
    """Iteratively remove all candidates with the lowest Borda score until a single 
    candidate remains.   If, at any stage, all  candidates have the same Borda score, 
    then all (remaining) candidates are winners.
    """

    num_cands = profile.num_cands   
    candidates = profile.candidates
    strict_maj_size = profile.strict_maj_size()
    
    rs, rcounts = profile.rankings_counts # get all the ranking data
    
    cands_to_ignore = np.empty(0)

    borda_scores = {c: _borda_score(rs, rcounts, num_cands, c) for c in candidates}

    min_borda_score = min(list(borda_scores.values()))
    
    last_place_borda_scores = [c for c in candidates 
                               if c in borda_scores.keys() and borda_scores[c] == min_borda_score]
      
    cands_to_ignore = np.concatenate((cands_to_ignore, last_place_borda_scores), axis=None)
    
    winners = list()
    if cands_to_ignore.shape[0] ==  num_cands: # call candidates have lowest Borda score
        winners = sorted(last_place_borda_scores)
    else: # remove the candidates with lowest Borda score
        updated_profile = _find_updated_profile(rs, cands_to_ignore, num_cands)
    while len(winners) == 0:
        borda_scores = {c: _borda_score(updated_profile, rcounts, num_cands - cands_to_ignore.shape[0], c) 
                        for c in candidates if not isin(cands_to_ignore,c)}
                
        min_borda_score = min(borda_scores.values())
        last_place_borda_scores = [c for c in borda_scores.keys() if borda_scores[c] == min_borda_score]
        
        cands_to_ignore = np.concatenate((cands_to_ignore, last_place_borda_scores), axis=None)
                
        if cands_to_ignore.shape[0] == num_cands: # removed all remaining candidates
            winners = sorted(last_place_borda_scores)
        elif num_cands - cands_to_ignore.shape[0] ==  1: # only one candidate remains
            winners = sorted([c for c in candidates if c not in cands_to_ignore])
        else: 
            updated_profile = _find_updated_profile(rs, cands_to_ignore, num_cands)
    return sorted(winners)


@vm_name("Baldwin")
def baldwin_with_data(profile):
    """Iteratively remove all candidates with the lowest Borda score until a single 
    candidate remains.   If, at any stage, all  candidates have the same Borda score, 
    then all (remaining) candidates are winners.
    """

    num_cands = profile.num_cands   
    candidates = profile.candidates
    strict_maj_size = profile.strict_maj_size()
    
    rs, rcounts = profile.rankings_counts # get all the ranking data
    
    cands_to_ignore = np.empty(0)
    
    elims_list = list()

    borda_scores = {c: _borda_score(rs, rcounts, num_cands, c) for c in candidates}

    min_borda_score = min(list(borda_scores.values()))
    
    last_place_borda_scores = [c for c in candidates 
                               if c in borda_scores.keys() and borda_scores[c] == min_borda_score]
     
    elims_list.append(last_place_borda_scores)
    cands_to_ignore = np.concatenate((cands_to_ignore, last_place_borda_scores), axis=None)
    
    winners = list()
    if cands_to_ignore.shape[0] ==  num_cands: # call candidates have lowest Borda score
        winners = sorted(last_place_borda_scores)
    else: # remove the candidates with lowest Borda score
        updated_profile = _find_updated_profile(rs, cands_to_ignore, num_cands)
    while len(winners) == 0:
        borda_scores = {c: _borda_score(updated_profile, rcounts, num_cands - cands_to_ignore.shape[0], c) 
                        for c in candidates if not isin(cands_to_ignore,c)}
                
        min_borda_score = min(borda_scores.values())
        last_place_borda_scores = [c for c in borda_scores.keys() if borda_scores[c] == min_borda_score]
        
        elims_list.append(last_place_borda_scores)
        cands_to_ignore = np.concatenate((cands_to_ignore, last_place_borda_scores), axis=None)
                
        if cands_to_ignore.shape[0] == num_cands: # removed all remaining candidates
            winners = sorted(last_place_borda_scores)
        elif num_cands - cands_to_ignore.shape[0] ==  1: # only one candidate remains
            winners = sorted([c for c in candidates if c not in cands_to_ignore])
        else: 
            updated_profile = _find_updated_profile(rs, cands_to_ignore, num_cands)
    return sorted(winners), elims_list



@vm_name("Baldwin TB")
def baldwin_tb(profile, tie_breaker=None):
    """Iteratively remove all candidates with the lowest Borda score until a single 
    candidate remains.   If, at any stage, all  candidates have the same Borda score, 
    then all (remaining) candidates are winners.

    The tie-breaking rule is any linear order (i.e., list) of the candidates.  The default rule 
    is to order the candidates as follows: 0,....,num_cands-1

    """
    
    # the tie_breaker is any linear order (i.e., list) of the candidates
    tb = tie_breaker if tie_breaker is not None else list(range(profile.num_cands))

    num_cands = profile.num_cands   
    candidates = profile.candidates
    strict_maj_size = profile.strict_maj_size()
    
    rs, rcounts = profile.rankings_counts # get all the ranking data
    
    cands_to_ignore = np.empty(0)

    borda_scores = {c: _borda_score(rs, rcounts, num_cands, c) for c in candidates}

    min_borda_score = min(list(borda_scores.values()))
    
    last_place_borda_scores = [c for c in candidates 
                               if c in borda_scores.keys() and borda_scores[c] == min_borda_score]
      
    cand_to_remove = last_place_borda_scores[0]
    for c in last_place_borda_scores[1:]: 
        if tb.index(c) < tb.index(cand_to_remove):
            cand_to_remove = c
    cands_to_ignore = np.concatenate((cands_to_ignore, np.array([cand_to_remove])), axis=None)
    
    winners = list()
    if cands_to_ignore.shape[0] ==  num_cands: # call candidates have lowest Borda score
        winners = sorted(last_place_borda_scores)
    else: # remove the candidates with lowest Borda score
        updated_profile = _find_updated_profile(rs, cands_to_ignore, num_cands)
    while len(winners) == 0:
        borda_scores = {c: _borda_score(updated_profile, rcounts, num_cands - cands_to_ignore.shape[0], c) 
                        for c in candidates if not isin(cands_to_ignore,c)}
                
        min_borda_score = min(borda_scores.values())
        last_place_borda_scores = [c for c in borda_scores.keys() if borda_scores[c] == min_borda_score]
        
        # select the candidate to remove using the tie-breaking rule (a linear order over the candidates)
        cand_to_remove = last_place_borda_scores[0]
        for c in last_place_borda_scores[1:]: 
            if tb.index(c) < tb.index(cand_to_remove):
                cand_to_remove = c
        
        cands_to_ignore = np.concatenate((cands_to_ignore, np.array([cand_to_remove])), axis=None)
                
        if cands_to_ignore.shape[0] == num_cands: # removed all remaining candidates
            winners = sorted(last_place_borda_scores)
        elif num_cands - cands_to_ignore.shape[0] ==  1: # only one candidate remains
            winners = sorted([c for c in candidates if c not in cands_to_ignore])
        else: 
            updated_profile = _find_updated_profile(rs, cands_to_ignore, num_cands)
    return sorted(winners)

@vm_name("Baldwin PUT")
def baldwin_put(profile):
    """Baldwin with parallel universe tie-breaking (PUT).  Apply the baldwin method with a tie-breaker
    for each possible linear order over the candidates. 
    
    Warning: This will take a long time on profiles with many candidates."""
    
    candidates = profile.candidates    
    cw = profile.condorcet_winner()
    
    winners = list() if cw is None else [cw]

    if len(winners) == 0:
        # run Coombs with tie-breaker for each permulation of candidates
        for tb in permutations(candidates):
            winners += baldwin_tb(profile, tie_breaker = tb) 

    return sorted(list(set(winners)))

# ####

@vm_name("Strict Nanson")
def strict_nanson(profile):
    """Iteratively remove all candidates with the  Borda score strictly below the average Borda score
    until one candidate remains.   If, at any stage, all  candidates have the same Borda score, 
    then all (remaining) candidates are winners.
    """

    num_cands = profile.num_cands   
    candidates = profile.candidates
    strict_maj_size = profile.strict_maj_size()
    
    rs, rcounts = profile.rankings_counts # get all the ranking data
    
    cands_to_ignore = np.empty(0)

    borda_scores = {c: _borda_score(rs, rcounts, num_cands, c) for c in candidates}
    
    avg_borda_score = np.mean(list(borda_scores.values()))
    below_borda_avg_candidates = np.array([c for c in borda_scores.keys() if borda_scores[c] < avg_borda_score])
    
    cands_to_ignore = np.concatenate((cands_to_ignore, below_borda_avg_candidates), axis=None)
    
    if cands_to_ignore.shape[0] == num_cands:  # all candidates have same Borda score
        winners = sorted(candidates)
    else: 
        winners = list()
        updated_profile = _find_updated_profile(rs, cands_to_ignore, num_cands)
    
    while len(winners) == 0: 
        
        borda_scores = {c: _borda_score(updated_profile, rcounts, num_cands - cands_to_ignore.shape[0], c) 
                        for c in candidates if not isin(cands_to_ignore,c)}
        avg_borda_scores = np.mean(list(borda_scores.values()))
    
        below_borda_avg_candidates = np.array([c for c in borda_scores.keys() 
                                               if borda_scores[c] < avg_borda_scores])
        
        cands_to_ignore = np.concatenate((cands_to_ignore, below_borda_avg_candidates), axis=None)
                
        if (below_borda_avg_candidates.shape[0] == 0) or ((num_cands - cands_to_ignore.shape[0]) == 1):
            winners = sorted([c for c in candidates if c not in cands_to_ignore])
        else:
            updated_profile = _find_updated_profile(rs, cands_to_ignore, num_cands)
            
    return winners


@vm_name("Strict Nanson")
def strict_nanson_with_data(profile):
    """Iteratively remove all candidates with the  Borda score strictly below the average Borda score
    until one candidate remains.   If, at any stage, all  candidates have the same Borda score, 
    then all (remaining) candidates are winners.
    """

    num_cands = profile.num_cands   
    candidates = profile.candidates
    strict_maj_size = profile.strict_maj_size()
    
    elims_list = list()
    rs, rcounts = profile.rankings_counts # get all the ranking data
    
    cands_to_ignore = np.empty(0)

    borda_scores = {c: _borda_score(rs, rcounts, num_cands, c) for c in candidates}
    
    avg_borda_score = np.mean(list(borda_scores.values()))
    below_borda_avg_candidates = np.array([c for c in borda_scores.keys() if borda_scores[c] < avg_borda_score])
    
    elims_list.append(list(below_borda_avg_candidates))
    cands_to_ignore = np.concatenate((cands_to_ignore, below_borda_avg_candidates), axis=None)
    
    if cands_to_ignore.shape[0] == num_cands:  # all candidates have same Borda score
        winners = sorted(candidates)
    else: 
        winners = list()
        updated_profile = _find_updated_profile(rs, cands_to_ignore, num_cands)
    
    while len(winners) == 0: 
        
        borda_scores = {c: _borda_score(updated_profile, rcounts, num_cands - cands_to_ignore.shape[0], c) 
                        for c in candidates if not isin(cands_to_ignore,c)}
        avg_borda_scores = np.mean(list(borda_scores.values()))
    
        below_borda_avg_candidates = np.array([c for c in borda_scores.keys() 
                                               if borda_scores[c] < avg_borda_scores])
        
        elims_list.append(list(below_borda_avg_candidates))
        cands_to_ignore = np.concatenate((cands_to_ignore, below_borda_avg_candidates), axis=None)
                
        if (below_borda_avg_candidates.shape[0] == 0) or ((num_cands - cands_to_ignore.shape[0]) == 1):
            winners = sorted([c for c in candidates if c not in cands_to_ignore])
        else:
            updated_profile = _find_updated_profile(rs, cands_to_ignore, num_cands)
            
    return winners, elims_list


@vm_name("Weak Nanson")
def weak_nanson(profile):
    """Iteratively remove all candidates with the  Borda score less than or equal to the average Borda score
    until one candidate remains.   If, at any stage, all  candidates have the same Borda score, 
    then all (remaining) candidates are winners.
    """

    num_cands = profile.num_cands   
    candidates = profile.candidates
    strict_maj_size = profile.strict_maj_size()
    
    rs, rcounts = profile.rankings_counts # get all the ranking data
    
    cands_to_ignore = np.empty(0)

    borda_scores = {c: _borda_score(rs, rcounts, num_cands, c) for c in candidates}
    
    avg_borda_score = np.mean(list(borda_scores.values()))
    below_borda_avg_candidates = np.array([c for c in borda_scores.keys() 
                                           if borda_scores[c] <= avg_borda_score])
    
    cands_to_ignore = np.concatenate((cands_to_ignore, below_borda_avg_candidates), axis=None)
    
    if cands_to_ignore.shape[0] == num_cands:  # all candidates have same Borda score
        winners = sorted(candidates)
    else: 
        winners = list()
        updated_profile = _find_updated_profile(rs, cands_to_ignore, num_cands)
    
    while len(winners) == 0: 
        
        borda_scores = {c: _borda_score(updated_profile, rcounts, num_cands - cands_to_ignore.shape[0], c) 
                        for c in candidates if not isin(cands_to_ignore,c)}
        avg_borda_scores = np.mean(list(borda_scores.values()))
    
        below_borda_avg_candidates = np.array([c for c in borda_scores.keys() 
                                               if borda_scores[c] <= avg_borda_scores])
        
        cands_to_ignore = np.concatenate((cands_to_ignore, below_borda_avg_candidates), axis=None)
                
        if (num_cands - cands_to_ignore.shape[0]) == 0:
            winners = sorted(below_borda_avg_candidates)
        elif (num_cands - cands_to_ignore.shape[0]) == 1:
            winners = sorted([c for c in candidates if c not in cands_to_ignore])
        else:
            updated_profile = _find_updated_profile(rs, cands_to_ignore, num_cands)
            
    return winners


@vm_name("Weak Nanson")
def weak_nanson_with_data(profile):
    """Iteratively remove all candidates with the  Borda score less than or equal to the average Borda score
    until one candidate remains.   If, at any stage, all  candidates have the same Borda score, 
    then all (remaining) candidates are winners.
    """

    num_cands = profile.num_cands   
    candidates = profile.candidates
    strict_maj_size = profile.strict_maj_size()
    elims_list = list()
    rs, rcounts = profile.rankings_counts # get all the ranking data
    
    cands_to_ignore = np.empty(0)

    borda_scores = {c: _borda_score(rs, rcounts, num_cands, c) for c in candidates}
    
    avg_borda_score = np.mean(list(borda_scores.values()))
    below_borda_avg_candidates = np.array([c for c in borda_scores.keys() 
                                           if borda_scores[c] <= avg_borda_score])
    
    elims_list.append(list(below_borda_avg_candidates))
    
    cands_to_ignore = np.concatenate((cands_to_ignore, below_borda_avg_candidates), axis=None)
    
    if cands_to_ignore.shape[0] == num_cands:  # all candidates have same Borda score
        winners = sorted(candidates)
    else: 
        winners = list()
        updated_profile = _find_updated_profile(rs, cands_to_ignore, num_cands)
    
    while len(winners) == 0: 
        
        borda_scores = {c: _borda_score(updated_profile, rcounts, num_cands - cands_to_ignore.shape[0], c) 
                        for c in candidates if not isin(cands_to_ignore,c)}
        avg_borda_scores = np.mean(list(borda_scores.values()))
    
        below_borda_avg_candidates = np.array([c for c in borda_scores.keys() 
                                               if borda_scores[c] <= avg_borda_scores])
        
        elims_list.append(list(below_borda_avg_candidates))
        cands_to_ignore = np.concatenate((cands_to_ignore, below_borda_avg_candidates), axis=None)
                
        if (num_cands - cands_to_ignore.shape[0]) == 0:
            winners = sorted(below_borda_avg_candidates)
        elif (num_cands - cands_to_ignore.shape[0]) == 1:
            winners = sorted([c for c in candidates if c not in cands_to_ignore])
        else:
            updated_profile = _find_updated_profile(rs, cands_to_ignore, num_cands)
            
    return winners, elims_list

@vm_name("Simplified Bucklin")
def simplified_bucklin(profile): 
    '''If a candidate has a strict majority of first-place votes, then that candidate is the winner. 
    If no such candidate exists, then check the candidates that are ranked first or second.  If a candidate 
    has a strict majority of first- or second-place voters, then that candidate is the winner. 
    If no such winner is found move on to the 3rd, 4th, etc. place votes'''
    
    strict_maj_size = profile.strict_maj_size()
    
    ranks = range(1, profile.num_cands + 1)
    
    cand_to_num_voters_rank = dict()
    for r in ranks:
        cand_to_num_voters_rank[r] = {c: profile.num_rank(c, r) 
                                for c in profile.candidates}
        cand_scores = {c:sum([cand_to_num_voters_rank[_r][c] for _r in cand_to_num_voters_rank.keys()]) 
                       for c in profile.candidates}
        if any([s >= strict_maj_size for s in cand_scores.values()]):
            break
    
    return sorted([c for c in profile.candidates if cand_scores[c] >= strict_maj_size])


## TODO FIX>..
@vm_name("Bucklin")
def bucklin(profile): 
    '''If a candidate has a strict majority of first-place votes, then that candidate is the winner. 
    If no such candidate exists, then check the candidates that are ranked first or second.  If a candidate 
    has a strict majority of first- or second-place voters, then that candidate is the winner. 
    If no such winner is found move on to the 3rd, 4th, etc. place votes'''
    
    strict_maj_size = profile.strict_maj_size()
    
    ranks = range(1, profile.num_cands + 1)
    
    cand_to_num_voters_rank = dict()
    for r in ranks:
        cand_to_num_voters_rank[r] = {c: profile.num_rank(c, r) 
                                for c in profile.candidates}
        cand_scores = {c:sum([cand_to_num_voters_rank[_r][c] for _r in cand_to_num_voters_rank.keys()]) 
                       for c in profile.candidates}
        if any([s >= strict_maj_size for s in cand_scores.values()]):
            break
    
    max_score = max(cand_scores.values())
    return sorted([c for c in profile.candidates if cand_scores[c] >= max_score])

# ###
# Majority Graph Invariant Methods
#
# For each method, there is a second version appended with "_mg" that assumes that the 
# input is a margin graph (represented as a networkx graph)
# ###


# ###
# Helper functions for reasoning about margin graphs
# ###

def generate_weak_margin_graph(profile):
    '''generate the weak weighted margin graph, where there is an edge if the margin is greater than or 
    equal to 0.'''
    mg = nx.DiGraph()
    candidates = profile.candidates
    mg.add_nodes_from(candidates)

    mg.add_weighted_edges_from([(c1,c2,profile.margin(c1,c2))  
           for c1 in candidates 
           for c2 in candidates if c1 != c2 if profile.margin(c1, c2) >= 0])
    return mg

# flatten a 2d list - turn a 2d list into a single list of items
flatten = lambda l: [item for sublist in l for item in sublist]

def has_cycle(mg):
    """true if the margin graph mg has a cycle"""
    try:
        cycles =  nx.find_cycle(mg)
    except:
        cycles = list()
    return len(cycles) != 0

def does_create_cycle(g, edge):
    '''return True if adding the edge to g create a cycle.
    it is assumed that edge is already in g'''
    source = edge[0]
    target = edge[1]
    for n in g.predecessors(source):
        if nx.has_path(g, target, n): 
            return True
    return False

def unbeaten_candidates(mg): 
    """the set of candidates with no incoming arrows, i.e., the 
    candidates that are unbeaten"""
    return [n for n in mg.nodes if mg.in_degree(n) == 0]

def find_condorcet_winner(mg): 
    """the Condorcet winner is the candidate with an edge to every other candidate"""
    return [n for n in mg.nodes if mg.out_degree(n) == len(mg.nodes) -  1]

def find_weak_condorcet_winners(mg):
    """weak condorcet winners are candidates with no incoming edges"""
    return unbeaten_candidates(mg)

def find_condorcet_losers(mg):
    """A Condorcet loser is the candidate with incoming edges from every other candidate"""
    
    # edge case: there is no Condorcet loser if there is only one node
    if len(mg.nodes) == 1:
        return []
    return [n for n in mg.nodes if mg.in_degree(n) == len(mg.nodes) - 1]

def is_majority_preferred(mg, c1, c2): 
    """true if c1 is majority preferred to c2, i.e., there is an edge from c1 to c2"""
    return mg.has_edge(c1, c2)

def is_tied(mg, c1, c2): 
    """true if there is no edge between c1 and c2"""    
    return not mg.has_edge(c1, c2) and not mg.has_edge(c2, c1)

@vm_name("Condorcet")
def condorcet(profile):
    """Return the Condorcet winner if one exists, otherwise return all the candidates"""
    
    cond_winner = profile.condorcet_winner()
    return [cond_winner] if cond_winner is not None else sorted(profile.candidates)

@vm_name("Condorcet")
def condorcet_mg(mg):
    """Return the Condorcet winner if one exists, otherwise return all the candidates"""
    
    cond_winner = find_condorcet_winner(mg)
    return cond_winner if len(cond_winner) > 0 else sorted(mg.nodes)

def copeland_scores_prof(prof, alpha=0.5):
    """Copeland alpha score of candidate c is: 1 point for every candidate c2 that c is majority 
    preferred to and alpha points for every candidate that c is tied with."""

    mg = prof.margin_graph()
    c_scores = {c: 0.0 for c in mg.nodes}
    for c in mg.nodes:
        for c2 in mg.nodes:
            if c != c2 and is_majority_preferred(mg, c, c2):
                c_scores[c] += 1.0
            if c != c2 and is_tied(mg, c, c2): 
                c_scores[c] += alpha
    return c_scores


@vm_name("Copeland")
def copeland(profile): 
    """The Copeland score for c is the number of candidates that c is majority preferred to 
    minus the number of candidates majority preferred to c.   The Copeland winners are the candidates
    with the max Copeland score."""
    
    candidates = profile.candidates
    copeland_scores = {c:len([1 for c2 in candidates if profile.margin(c,c2) > 0]) - 
                       len([1 for c2 in candidates if profile.margin(c,c2) < 0]) 
                       for c in candidates}
    max_copeland_score = max(copeland_scores.values())
    return sorted([c for c in candidates if copeland_scores[c] == max_copeland_score])

@vm_name("Llull")
def llull(profile): 
    """The Llull score for a candidate c is the number of candidates that c is weakly majority 
    preferred to.   The Llull winners are the candidates with the greatest Llull score."""
    
    candidates = profile.candidates
    llull_scores = {c:len([1 for c2 in candidates if profile.margin(c,c2) >= 0])
                    for c in candidates}
    max_llull_score = max(llull_scores.values())
    return sorted([c for c in candidates if llull_scores[c] == max_llull_score])


######
# Copeland/Llull on margin graphs
######
def copeland_scores(mg, alpha=0.5):
    """Copeland alpha score of candidate c is: 1 point for every candidate c2 that c is majority 
    preferred to and alpha points for every candidate that c is tied with."""
    c_scores = {c: 0.0 for c in mg.nodes}
    for c in mg.nodes:
        for c2 in mg.nodes:
            if c != c2 and is_majority_preferred(mg, c, c2):
                c_scores[c] += 1.0
            if c != c2 and is_tied(mg, c, c2): 
                c_scores[c] += alpha
    return c_scores

@vm_name("Copeland")    
def copeland_mg(mg): 
    """Copeland winners are the candidates with maximum Copeland_alpha socre with alpha=0.5"""
    c_scores = copeland_scores(mg)
    max_score = max(c_scores.values())
    return sorted([c for c in c_scores.keys() if c_scores[c] == max_score])

@vm_name("Llull")    
def llull_mg(mg): 
    """Llull winners are the candidates with maximum Copeland_alpha socre with alpha=1"""
    c_scores = copeland_scores(mg, alpha=1)
    max_score = max(c_scores.values())
    return sorted([c for c in c_scores.keys() if c_scores[c] == max_score])


# # Uncovered Set

def left_covers(dom, c1, c2):
    # left covers: c1 left covers c2 when all the candidates that are majority preferred to c1
    # are also majority preferred to c2.
    
    # weakly left covers: c1 weakly left covers c2 when all the candidates that are majority preferred to or tied with c1
    # are also majority preferred to or tied with c2.
    
    return dom[c1].issubset(dom[c2])

def right_covers(dom, c1, c2):
    # right covers: c1 right covers c2 when all the candidates that c2  majority preferrs are majority
    # preferred by c1
      
    return dom[c2].issubset(dom[c1])


@vm_name("Uncovered Set")
def uc_gill_mg(mg): 
    """(Gillies version)   Given candidates a and b, say that a defeats b in the profile P, a defeats b 
    if a is majority preferred to b and a left covers b: i.e., for all c, if c is majority preferred to a, 
    then c majority preferred to b. Then the winners are the set of  candidates who are undefeated in P. 
    """
    
    dom = {n: set(mg.predecessors(n)) for n in mg.nodes}
    uc_set = list()
    for c1 in mg.nodes:
        is_in_ucs = True
        for c2 in mg.predecessors(c1): # consider only c2 predecessors
            if c1 != c2:
                # check if c2 left covers  c1 
                if left_covers(dom, c2, c1):
                    is_in_ucs = False
        if is_in_ucs:
            uc_set.append(c1)
    return list(sorted(uc_set))

@vm_name("Uncovered Set")
def uc_gill(profile): 
    '''See the explanation for uc_gill_mg'''
    
    mg = profile.margin_graph() 
    return uc_gill_mg(mg)

@vm_name("Uncovered Set - Fishburn")
def uc_fish_mg(mg): 
    """(Fishburn version)  Given candidates a and b, say that a defeats b in the profile P
    if a left covers b: i.e., for all c, if c is majority preferred to a, then c majority preferred to b, and
    b does not left cover a. Then the winners are the set of candidates who are undefeated."""
    
    dom = {n: set(mg.predecessors(n)) for n in mg.nodes}
    uc_set = list()
    for c1 in mg.nodes:
        is_in_ucs = True
        for c2 in mg.nodes:
            if c1 != c2:
                # check if c2 left covers  c1 but c1 does not left cover c2
                if left_covers(dom, c2, c1)  and not left_covers(dom, c1, c2):
                    is_in_ucs = False
        if is_in_ucs:
            uc_set.append(c1)
    return list(sorted(uc_set))

@vm_name("UC Fishburn")
def uc_fish(profile): 
    """See the explaination of uc_fish_mg"""
    
    mg = profile.margin_graph() 
    return uc_fish_mg(mg)


@vm_name("UC Bordes")
def uc_bordes_mg(mg): 
    """Bordes version: a Bordes covers b if a is majority preferred to b and for all c, if c is 
    majority preferred or tied with a, then c is majority preferred to tied with b. Returns the candidates
    that are not Bordes covered. 

    """

    dom = {n: set(mg.predecessors(n)).union([_n for _n in mg.nodes 
                                             if (not mg.has_edge(n, _n) and not mg.has_edge(_n, n))]) 
           for n in mg.nodes}
    
    uc_set = list()
    for c1 in mg.nodes:
        is_in_ucs = True
        for c2 in mg.predecessors(c1): # consider only c2 predecessors
            if c1 != c2:
                # check if c2 left covers  c1 
                if left_covers(dom, c2, c1):
                    is_in_ucs = False
        if is_in_ucs:
            uc_set.append(c1)
    return list(sorted(uc_set))  

@vm_name("UC Bordes")
def uc_bordes(profile): 
    '''
    See explanation for uc_bordes_mg 
    '''
    mg = profile.margin_graph() 
    return uc_bordes_mg(mg)

@vm_name("UC McKelvey")
def uc_mckelvey_mg(mg): 
    """McKelvey version: a McKelvey covers b if a Gillies covers b and a Bordes covers b. Returns the candidates
    that are not McKelvey covered. 

    """
    weak_dom = {n: set(mg.predecessors(n)).union([_n for _n in mg.nodes 
                                                  if (not mg.has_edge(n, _n) and not mg.has_edge(_n, n))]) 
                for n in mg.nodes}
    strict_dom = {n: set(mg.predecessors(n)) for n in mg.nodes}    
    uc_set = list()
    for c1 in mg.nodes:
        is_in_ucs = True
        for c2 in mg.predecessors(c1): # consider only c2 predecessors
            if c1 != c2:
                # check if c2 left covers  c1 
                if left_covers(strict_dom, c2, c1) and left_covers(weak_dom, c2, c1):
                    is_in_ucs = False
        if is_in_ucs:
            uc_set.append(c1)
    return list(sorted(uc_set))      

@vm_name("UC McKelvey")
def uc_mckelvey(profile): 
    mg = profile.margin_graph() 
    return uc_mckelvey_mg(mg)

# ## Same as UC Bordes
# @vm_name("UC Right Covers")
# def uc_right_covers_mg(mg): 
#    """Right covers
#
#    """
#    
#    dom = {n: set(mg.successors(n)) for n in mg.nodes}
#    
#    uc_set = list()
#    for c1 in mg.nodes:
#        is_in_ucs = True
#        for c2 in mg.predecessors(c1): # consider only c2 predecessors
#            if c1 != c2:
#                # check if c2 right covers  c1 
#                if right_covers(dom, c2, c1):
#                    is_in_ucs = False
#        if is_in_ucs:
#            uc_set.append(c1)
#    return list(sorted(uc_set))    
#   
# @vm_name("UC Right Covers")
# def uc_right_covers(profile): 
#    mg = profile.margin_graph() 
#    return uc_right_covers_mg(mg)
#  
# ####

@vm_name("Top Cycle")
def getcha_mg(mg):
    """The smallest set of candidates such that every candidate inside the set 
    is majority preferred to every candidate outside the set.  Also known as GETCHA or the Smith set.
    """
    min_indegree = min([max([mg.in_degree(n) for n in comp]) for comp in nx.strongly_connected_components(mg)])
    smith = [comp for comp in nx.strongly_connected_components(mg) if max([mg.in_degree(n) for n in comp]) == min_indegree][0]
    return sorted(list(smith))

@vm_name("Top Cycle")
def getcha(profile):
    """See the explanation of getcha_mg"""
    mg = generate_weak_margin_graph(profile)
    return getcha_mg(mg)

@vm_name("GOCHA")
def gocha_mg(mg):
    """The GOCHA set (also known as the Schwartz set) is the smallest set of candidates with the property
    that every candidate inside the set is not majority preferred by every candidate outside the set. 
    """
    transitive_closure =  nx.algorithms.dag.transitive_closure(mg)
    schwartz = set()
    for ssc in nx.strongly_connected_components(transitive_closure):
        if not any([transitive_closure.has_edge(c2,c1) 
                    for c1 in ssc for c2 in transitive_closure.nodes if c2 not in ssc]):
            schwartz =  schwartz.union(ssc)
    return sorted(list(schwartz))

@vm_name("GOCHA")
def gocha(profile):
    """See the explanation of gocha_mg""" 
    mg = profile.margin_graph()
    return gocha_mg(mg)


# ####
# (Qualitative) Margin Graph invariant methods
# ####

def minimax_scores(profile, score_method="margins"):
    """Return the minimax scores for each candidate, where the minimax score for c in 
    the smallest maximum pairwise defeat. 
    """
    
    candidates = profile.candidates
    if len(candidates) == 1:
        return {c: 0 for c in candidates}
    
    # there are different scoring functions that can be used to measure the worse loss for each 
    # candidate. These all produce the same set of winners when voters submit linear orders. 
    score_functions = {
        "winning": lambda c1,c2: profile.support(c1,c2) if profile.support(c1,c2) > profile.support(c2,c1) else 0,
        "margins": lambda c1,c2: profile.support(c1,c2)   -  profile.support(c2,c1),
        "pairwise_opposition": lambda c1,c2: profile.support(c1,c2)
    } 
    dominators = {c : [_c for _c in candidates if _c != c and profile.margin(_c, c)> 0] for c in candidates}
    scores = {c: -max([score_functions[score_method](_c,c) for _c in dominators[c]]) if len(dominators[c]) > 0 else 0 for c in candidates}
    return scores


@vm_name("Minimax")
def minimax(profile, score_method="margins"):
    """Return the candidates with the smallest maximum pairwise defeat.  That is, for each 
    candidate c determine the biggest margin of a candidate c1 over c, then select 
    the candidates with the smallest such loss. Alson known as the Simpson-Kramer Rule.
    """
    
    candidates = profile.candidates
    
    if len(candidates) == 1:
        return candidates
    
    # there are different scoring functions that can be used to measure the worse loss for each 
    # candidate. These all produce the same set of winners when voters submit linear orders. 
    score_functions = {
        "winning": lambda c1,c2: profile.support(c1,c2) if profile.support(c1,c2) > profile.support(c2,c1) else 0,
        "margins": lambda c1,c2: profile.support(c1,c2)   -  profile.support(c2,c1),
        "pairwise_opposition": lambda c1,c2: profile.support(c1,c2)
    } 
    scores = {c: max([score_functions[score_method](_c,c) for _c in candidates if _c != c]) 
              for c in candidates}
    min_score = min(scores.values())
    return sorted([c for c in candidates if scores[c] == min_score])

@vm_name("Minimax")
def minimax_mg(mg):
    
    max_losses = {c : max([mg[d][c]['weight'] for d in mg.nodes if mg.has_edge(d,c)]) if mg.in_degree(c) > 0 else 0 for c in mg.nodes}
    min_max_loss = min(max_losses.values())
    return sorted([c for c in mg.nodes if max_losses[c] == min_max_loss])

def cycle_number(profile):
    
    candidates = profile.candidates 
    
    # create the margin graph
    mg = profile.margin_graph()
    
    # find the cycle number for each candidate
    cycle_number = {cs:0 for cs in permutations(candidates,2)}
    for cycle in nx.simple_cycles(mg): # for each cycle in the margin graph

        # get all the margins (i.e., the weights) of the edges in the cycle
        margins = list() 
        for idx,c1 in enumerate(cycle): 
            next_idx = idx + 1 if (idx + 1) < len(cycle) else 0
            c2 = cycle[next_idx]
            margins.append(mg[c1][c2]['weight'])
            
        split_number = min(margins) # the split number of the cycle is the minimal margin
        for c1,c2 in cycle_number.keys():
            c1_index = cycle.index(c1) if c1 in cycle else -1
            c2_index = cycle.index(c2) if c2 in cycle else -1

            # only need to check cycles with an edge from c1 to c2
            if (c1_index != -1 and c2_index != -1) and ((c2_index == c1_index + 1) or (c1_index == len(cycle)-1 and c2_index == 0)):
                cycle_number[(c1,c2)] = split_number if split_number > cycle_number[(c1,c2)] else cycle_number[(c1,c2)]        
    return cycle_number

def split_cycle_defeat(profile):
    """A majority cycle in a profile P is a sequence x_1,...,x_n of distinct candidates in 
    P with x_1=x_n such that for 1 <= k <= n-1,  x_k is majority preferred to x_{k+1}.
    The *strength of* a majority is the minimal margin in the cycle.  
    Say that a defeats b in P if the margin of a over b is positive and greater than 
    the strength of the strongest majority cycle containing a and b. The Split Cycle winners
    are the undefeated candidates.
    """
    
    candidates = profile.candidates 
    
    # create the margin graph
    mg = profile.margin_graph()
    
    # find the cycle number for each candidate
    cycle_number = {cs:0 for cs in permutations(candidates,2)}
    for cycle in nx.simple_cycles(mg): # for each cycle in the margin graph

        # get all the margins (i.e., the weights) of the edges in the cycle
        margins = list() 
        for idx,c1 in enumerate(cycle): 
            next_idx = idx + 1 if (idx + 1) < len(cycle) else 0
            c2 = cycle[next_idx]
            margins.append(mg[c1][c2]['weight'])
            
        split_number = min(margins) # the split number of the cycle is the minimal margin
        for c1,c2 in cycle_number.keys():
            c1_index = cycle.index(c1) if c1 in cycle else -1
            c2_index = cycle.index(c2) if c2 in cycle else -1

            # only need to check cycles with an edge from c1 to c2
            if (c1_index != -1 and c2_index != -1) and ((c2_index == c1_index + 1) or (c1_index == len(cycle)-1 and c2_index == 0)):
                cycle_number[(c1,c2)] = split_number if split_number > cycle_number[(c1,c2)] else cycle_number[(c1,c2)]        

    # construct the defeat relation, where a defeats b if margin(a,b) > cycle_number(a,b) (see Lemma 3.13)
    defeat = nx.DiGraph()
    defeat.add_nodes_from(candidates)
    defeat.add_weighted_edges_from([(c1,c2, profile.margin(c1, c2))  
           for c1 in candidates 
           for c2 in candidates if c1 != c2 if profile.margin(c1,c2) > cycle_number[(c1,c2)]])

    return defeat

def split_cycle_defeat_mg(mg):
    """A majority cycle in a profile P is a sequence x_1,...,x_n of distinct candidates in 
    P with x_1=x_n such that for 1 <= k <= n-1,  x_k is majority preferred to x_{k+1}.
    The *strength of* a majority is the minimal margin in the cycle.  
    Say that a defeats b in P if the margin of a over b is positive and greater than 
    the strength of the strongest majority cycle containing a and b. The Split Cycle winners
    are the undefeated candidates.
    """
    
    candidates = mg.nodes 
    
    
    # find the cycle number for each candidate
    cycle_number = {cs:0 for cs in permutations(candidates,2)}
    for cycle in nx.simple_cycles(mg): # for each cycle in the margin graph

        # get all the margins (i.e., the weights) of the edges in the cycle
        margins = list() 
        for idx,c1 in enumerate(cycle): 
            next_idx = idx + 1 if (idx + 1) < len(cycle) else 0
            c2 = cycle[next_idx]
            margins.append(mg[c1][c2]['weight'])
            
        split_number = min(margins) # the split number of the cycle is the minimal margin
        for c1,c2 in cycle_number.keys():
            c1_index = cycle.index(c1) if c1 in cycle else -1
            c2_index = cycle.index(c2) if c2 in cycle else -1

            # only need to check cycles with an edge from c1 to c2
            if (c1_index != -1 and c2_index != -1) and ((c2_index == c1_index + 1) or (c1_index == len(cycle)-1 and c2_index == 0)):
                cycle_number[(c1,c2)] = split_number if split_number > cycle_number[(c1,c2)] else cycle_number[(c1,c2)]        

    # construct the defeat relation, where a defeats b if margin(a,b) > cycle_number(a,b) (see Lemma 3.13)
    defeat = nx.DiGraph()
    defeat.add_nodes_from(candidates)
    defeat.add_weighted_edges_from([(c1,c2, mg[c1][c2]['weight'])  
           for c1 in candidates 
           for c2 in candidates if c1 != c2 if mg.has_edge(c1,c2) and mg[c1][c2]['weight'] > cycle_number[(c1,c2)]])

    return defeat


@vm_name("Split Cycle") 
def split_cycle_mg(mg):
    """A *majority cycle in a profile P is a sequence x_1,...,x_n of distinct candidates in 
    P with x_1=x_n such that for 1 <= k <= n-1,  x_k is majority preferred to x_{k+1}.
    The *strength of* a majority is the minimal margin in the cycle.  
    Say that a defeats b in P if the margin of a over b is positive and greater than 
    the strength of the strongest majority cycle containing a and b. The Split Cycle winners
    are the undefeated candidates.
    """
    
    candidates = mg.nodes 
    # find the cycle number for each candidate
    cycle_number = {cs:0 for cs in permutations(candidates,2)}
    for cycle in nx.simple_cycles(mg): # for each cycle in the margin graph

        # get all the margins (i.e., the weights) of the edges in the cycle
        margins = list() 
        for idx,c1 in enumerate(cycle): 
            next_idx = idx + 1 if (idx + 1) < len(cycle) else 0
            c2 = cycle[next_idx]
            margins.append(mg[c1][c2]['weight'])
            
        split_number = min(margins) # the split number of the cycle is the minimal margin
        for c1,c2 in cycle_number.keys():
            c1_index = cycle.index(c1) if c1 in cycle else -1
            c2_index = cycle.index(c2) if c2 in cycle else -1

            # only need to check cycles with an edge from c1 to c2
            if (c1_index != -1 and c2_index != -1) and ((c2_index == c1_index + 1) or (c1_index == len(cycle)-1 and c2_index == 0)):
                cycle_number[(c1,c2)] = split_number if split_number > cycle_number[(c1,c2)] else cycle_number[(c1,c2)]        

    # construct the defeat relation, where a defeats b if margin(a,b) > cycle_number(a,b) (see Lemma 3.13)
    defeat = nx.DiGraph()
    defeat.add_nodes_from(candidates)
    defeat.add_edges_from([(c1,c2)  
           for c1 in candidates 
           for c2 in candidates if c1 != c2 if mg.has_edge(c1, c2) and mg[c1][c2]['weight'] > cycle_number[(c1,c2)]])

    # the winners are candidates not defeated by any other candidate
    winners = unbeaten_candidates(defeat)
    
    return sorted(list(set(winners)))


@vm_name("Split Cycle") 
def split_cycle(profile):
    """A *majority cycle in a profile P is a sequence x_1,...,x_n of distinct candidates in 
    P with x_1=x_n such that for 1 <= k <= n-1,  x_k is majority preferred to x_{k+1}.
    The *strength of* a majority is the minimal margin in the cycle.  
    Say that a defeats b in P if the margin of a over b is positive and greater than 
    the strength of the strongest majority cycle containing a and b. The Split Cycle winners
    are the undefeated candidates.
    """
    
    candidates = profile.candidates 
    
    # create the margin graph
    mg = profile.margin_graph()
    
    # find the cycle number for each candidate
    cycle_number = {cs:0 for cs in permutations(candidates,2)}
    for cycle in nx.simple_cycles(mg): # for each cycle in the margin graph

        # get all the margins (i.e., the weights) of the edges in the cycle
        margins = list() 
        for idx,c1 in enumerate(cycle): 
            next_idx = idx + 1 if (idx + 1) < len(cycle) else 0
            c2 = cycle[next_idx]
            margins.append(mg[c1][c2]['weight'])
            
        split_number = min(margins) # the split number of the cycle is the minimal margin
        for c1,c2 in cycle_number.keys():
            c1_index = cycle.index(c1) if c1 in cycle else -1
            c2_index = cycle.index(c2) if c2 in cycle else -1

            # only need to check cycles with an edge from c1 to c2
            if (c1_index != -1 and c2_index != -1) and ((c2_index == c1_index + 1) or (c1_index == len(cycle)-1 and c2_index == 0)):
                cycle_number[(c1,c2)] = split_number if split_number > cycle_number[(c1,c2)] else cycle_number[(c1,c2)]        

    # construct the defeat relation, where a defeats b if margin(a,b) > cycle_number(a,b) (see Lemma 3.13)
    defeat = nx.DiGraph()
    defeat.add_nodes_from(candidates)
    defeat.add_edges_from([(c1,c2)  
           for c1 in candidates 
           for c2 in candidates if c1 != c2 if profile.margin(c1,c2) > cycle_number[(c1,c2)]])

    # the winners are candidates not defeated by any other candidate
    winners = unbeaten_candidates(defeat)
    
    return sorted(list(set(winners)))

@vm_name("Split Cycle")
def split_cycle_faster(profile):   
    """Implementation of Split Cycle using a variation of the Floyd Warshall-Algorithm  
    """
    candidates = profile.candidates
    weak_condorcet_winners = {c:True for c in candidates}
    mg = [[-np.inf for _ in candidates] for _ in candidates]
    
    # Weak Condorcet winners are Split Cycle winners
    for c1 in candidates:
        for c2 in candidates:
            if (profile.support(c1,c2) > profile.support(c2,c1) or c1 == c2):
                mg[c1][c2] = profile.support(c1,c2) - profile.support(c2,c1)
                weak_condorcet_winners[c2] = weak_condorcet_winners[c2] and (c1 == c2)
    
    strength = list(map(lambda i : list(map(lambda j : j , i)) , mg))
    for i in candidates:         
        for j in candidates: 
            if i!= j:
                if not weak_condorcet_winners[j]: # weak Condorcet winners are Split Cycle winners
                    for k in candidates: 
                        if i!= k and j != k:
                            strength[j][k] = max(strength[j][k], min(strength[j][i],strength[i][k]))
    winners = {i:True for i in candidates}
    for i in candidates: 
        for j in candidates:
            if i!=j:
                if mg[j][i] > strength[i][j]: # the main difference with Beat Path
                    winners[i] = False
    return sorted([c for c in candidates if winners[c]])

@vm_name("Split Cycle")
def split_cycle_faster_mg(mg):   
    """Implementation of Split Cycle using a variation of the Floyd Warshall-Algorithm  
    """
    cmap = {cidx : c for cidx, c in enumerate(mg.nodes)}
    candidates = mg.nodes
    weak_condorcet_winners = {c:True for c in candidates}
    m_matrix = [[-np.inf for _ in candidates] for _ in candidates]
    
    # Weak Condorcet winners are Split Cycle winners
    for c1 in candidates:
        for c2 in candidates:
            if (mg.has_edge(c1,c2) or c1 == c2):
                m_matrix[c1][c2] = mg[c1][c2]['weight'] if c1 != c2 else 0
                weak_condorcet_winners[c2] = weak_condorcet_winners[c2] and (c1 == c2)
    
    strength = list(map(lambda i : list(map(lambda j : j , i)) , m_matrix))
    for i in candidates:         
        for j in candidates: 
            if i!= j:
                if not weak_condorcet_winners[j]: # weak Condorcet winners are Split Cycle winners
                    for k in candidates: 
                        if i!= k and j != k:
                            strength[j][k] = max(strength[j][k], min(strength[j][i],strength[i][k]))
    winners = {i:True for i in candidates}
    for i in candidates: 
        for j in candidates:
            if i!=j:
                if m_matrix[j][i] > strength[i][j]: # the main difference with Beat Path
                    winners[i] = False
    return sorted([c for c in candidates if winners[c]])

@vm_name("Iterated Split Cycle")
def iterated_splitcycle(prof):
    '''Iteratively calculate the split cycle winners until there is a
    unique winner or all remaining candidates are split cycle winners'''
    

    sc_winners = split_cycle_faster(prof)
    orig_cnames = {c:c for c in prof.candidates}
    
    reduced_prof = prof
    
    while len(sc_winners) != 1 and sc_winners != list(reduced_prof.candidates): 
        reduced_prof, orig_cnames = prof.remove_candidates([c for c in prof.candidates if c not in sc_winners])
        sc_winners = split_cycle_faster(reduced_prof)
        
    return sorted([orig_cnames[c] for c in sc_winners])

@vm_name("Beat Path")
def beat_path(profile): 
    """For candidates a and b, a *path from a to b in P* is a sequence 
    x_1,...,x_n of distinct candidates in P with  x_1=a and x_n=b such that 
    for 1 <= k <= n-1$, x_k is majority preferred to x_{k+1}.  The *strength of a path* 
    is the minimal margin along that path.  Say that a defeats b in P if 
    the strength of the strongest path from a to b is greater than the strength of 
    the strongest path from b to a. Then Beat Path winners are the undefeated candidates. 
    Also known as the Schulze Rule.
    """
    
    #1. calculate vote_graph, edge from c1 to c2 of c1 beats c2, weighted by support for c1 over c2
    #2. For all pairs c1, c2, find all paths from c1 to c2, for each path find the minimum weight.  
    #   beatpath[c1,c2] = max(weight(p) all p's from c1 to c2)
    #3. winner is the candidates that beat every other candidate 

    candidates = profile.candidates
    mg = profile.margin_graph()
    beat_paths_weights = {c: {c2:0 for c2 in candidates if c2 != c} for c in candidates}
    for c in candidates: 
        for other_c in beat_paths_weights[c].keys():
            all_paths =  list(nx.all_simple_paths(mg, c, other_c))
            if len(all_paths) > 0:
                beat_paths_weights[c][other_c] = max([min([mg[p[i]][p[i+1]]['weight'] 
                                                           for i in range(0,len(p)-1)]) 
                                                      for p in all_paths])
    
    winners = list()
    for c in candidates: 
        if all([beat_paths_weights[c][c2] >= beat_paths_weights[c2][c] 
                for c2 in candidates  if c2 != c]):
            winners.append(c)

    return sorted(list(winners))

@vm_name("Beat Path")
def beat_path_faster(profile):   
    """Implementation of Beat Path using a variation of the Floyd Warshall-Algorithm
    See https://en.wikipedia.org/wiki/Schulze_method#Implementation
    """
    
    candidates = profile.candidates
    
    mg = [[-np.inf for _ in candidates] for _ in candidates]
    for c1 in candidates:
        for c2 in candidates:
            if (profile.support(c1,c2) > profile.support(c2,c1) or c1 == c2):
                mg[c1][c2] = profile.support(c1,c2) - profile.support(c2,c1)
    strength = list(map(lambda i : list(map(lambda j : j , i)) , mg))
    for i in candidates:         
        for j in candidates: 
            if i!= j:
                for k in candidates: 
                    if i!= k and j != k:
                        strength[j][k] = max(strength[j][k], min(strength[j][i],strength[i][k]))
    winners = {i:True for i in candidates}
    for i in candidates: 
        for j in candidates:
            if i!=j:
                if strength[j][i] > strength[i][j]:
                    winners[i] = False
    return sorted([c for c in candidates if winners[c]])


@vm_name("Beat Path")
def beat_path_mg(mg): 
    """For candidates a and b, a *path from a to b in P* is a sequence 
    x_1,...,x_n of distinct candidates in P with  x_1=a and x_n=b such that 
    for 1 <= k <= n-1$, x_k is majority preferred to x_{k+1}.  The *strength of a path* 
    is the minimal margin along that path.  Say that a defeats b in P if 
    the strength of the strongest path from a to b is greater than the strength of 
    the strongest path from b to a. Then Beat Path winners are the undefeated candidates. 
    Also known as the Schulze Rule.
    """
    
    #1. calculate vote_graph, edge from c1 to c2 of c1 beats c2, weighted by support for c1 over c2
    #2. For all pairs c1, c2, find all paths from c1 to c2, for each path find the minimum weight.  
    #   beatpath[c1,c2] = max(weight(p) all p's from c1 to c2)
    #3. winner is the candidates that beat every other candidate 

    candidates = mg.nodes
    
    beat_paths_weights = {c: {c2:0 for c2 in candidates if c2 != c} for c in candidates}
    for c in candidates: 
        for other_c in beat_paths_weights[c].keys():
            all_paths =  list(nx.all_simple_paths(mg, c, other_c))
            if len(all_paths) > 0:
                beat_paths_weights[c][other_c] = max([min([mg[p[i]][p[i+1]]['weight'] 
                                                           for i in range(0,len(p)-1)]) 
                                                      for p in all_paths])
    
    winners = list()
    for c in candidates: 
        if all([beat_paths_weights[c][c2] >= beat_paths_weights[c2][c] 
                for c2 in candidates  if c2 != c]):
            winners.append(c)

    return sorted(list(winners))

@vm_name("Beat Path")
def beat_path_faster_mg(mg):   
    """Implementation of Beat Path using a variation of the Floyd Warshall-Algorithm
    See https://en.wikipedia.org/wiki/Schulze_method#Implementation
    """
    
    candidates = mg.nodes
    
    m_matrix = [[-np.inf for _ in candidates] for _ in candidates]
    for c1 in candidates:
        for c2 in candidates:
            if (mg.has_edge(c1,c2) or c1 == c2):
                m_matrix[c1][c2] = mg[c1][c2]['weight'] if c1 != c2 else 0
    strength = list(map(lambda i : list(map(lambda j : j , i)) , m_matrix))
    for i in candidates:         
        for j in candidates: 
            if i!= j:
                for k in candidates: 
                    if i!= k and j != k:
                        strength[j][k] = max(strength[j][k], min(strength[j][i],strength[i][k]))
    winners = {i:True for i in candidates}
    for i in candidates: 
        for j in candidates:
            if i!=j:
                if strength[j][i] > strength[i][j]:
                    winners[i] = False
    return sorted([c for c in candidates if winners[c]])


@vm_name("Ranked Pairs")
def ranked_pairs(profile):
    """Order the edges in the weak margin graph from largest to smallest and lock them 
    in in that order, skipping edges that create a cycle.   Break ties using a tie-breaking 
    linear ordering over the edges.   A candidate is a Ranked Pairs winner if it wins according 
    to some tie-breaking rule. Also known as Tideman's Rule.
    """
    wmg = generate_weak_margin_graph(profile)
    cw = profile.condorcet_winner()
    # Ranked Pairs is Condorcet consistent, so simply return the Condorcet winner if exists
    if cw is not None: 
        winners = [cw]
    else:
        winners = list()            
        margins = sorted(list(set([e[2]['weight'] for e in wmg.edges(data=True)])), reverse=True)
        sorted_edges = [[e for e in wmg.edges(data=True) if e[2]['weight'] == m] for m in margins]
        tbs = product(*[permutations(edges) for edges in sorted_edges])
        for tb in tbs:
            edges = flatten(tb)
            new_ranking = nx.DiGraph() 
            for e in edges: 
                new_ranking.add_edge(e[0], e[1], weight=e[2]['weight'])
                if does_create_cycle(new_ranking, e):
                    new_ranking.remove_edge(e[0], e[1])
            winners.append(unbeaten_candidates(new_ranking)[0])
    return sorted(list(set(winners)))

@vm_name("Ranked Pairs ZT")
def ranked_pairs_zt(profile):
    '''Ranked pairs (see the ranked_pairs docstring for an explanation) where a fixed voter breaks 
    any ties in the margins.  It is always the voter in position 0 that breaks the ties. 
    Since voters have strict preferences, this method is resolute.  This is known as Ranked Pairs ZT, 
    for Zavist Tideman 
    '''
    # the tie-breaker is always the first voter: 
    tb_ranking = tuple(profile._rankings[0])
    
    wmg = generate_weak_margin_graph(profile)
    cw = profile.condorcet_winner()

    # Ranked Pairs is Condorcet consistent, so simply return the Condorcet winner if exists
    if cw is not None: 
        winners = [cw]
    else:
        winners = list() 
        margins = sorted(list(set([e[2]['weight'] for e in wmg.edges(data=True)])), reverse=True)
        rp_defeat = nx.DiGraph() 
        for m in margins: 
            edges = [e for e in wmg.edges(data=True) if e[2]['weight'] == m]
            
            # break ties using the lexicgraphic ordering on tuples given tb_ranking
            sorted_edges = sorted(edges, key = lambda e: (tb_ranking.index(e[0]), tb_ranking.index(e[1])), reverse=False)
            for e in sorted_edges: 
                rp_defeat.add_edge(e[0], e[1], weight=e[2]['weight'])
                if  does_create_cycle(rp_defeat, e):
                    rp_defeat.remove_edge(e[0], e[1])
        winners = unbeaten_candidates(rp_defeat)
    return sorted(list(set(winners)))

@vm_name("Ranked Pairs T")
def ranked_pairs_t(profile, tiebreaker = None):
    '''Ranked pairs (see the ranked_pairs docstring for an explanation) where a fixed linear order on the 
    candidates to break any ties in the margins.  It is always the voter in position 0 that breaks the ties. 
    Since voters have strict preferences, this method is resolute.
    '''
    
    tb_ranking = tiebreaker if tiebreaker is not None else sorted(list(profile.candidates))
    wmg = generate_weak_margin_graph(profile)
    cw = profile.condorcet_winner()

    # Ranked Pairs is Condorcet consistent, so simply return the Condorcet winner if exists
    if cw is not None: 
        winners = [cw]
    else:
        winners = list() 
        margins = sorted(list(set([e[2]['weight'] for e in wmg.edges(data=True)])), reverse=True)
        rp_defeat = nx.DiGraph() 
        for m in margins: 
            edges = [e for e in wmg.edges(data=True) if e[2]['weight'] == m]
            
            # break ties using the lexicgraphic ordering on tuples given tb_ranking
            sorted_edges = sorted(edges, key = lambda e: (tb_ranking.index(e[0]), tb_ranking.index(e[1])), reverse=False)
            for e in sorted_edges: 
                rp_defeat.add_edge(e[0], e[1], weight=e[2]['weight'])
                if  does_create_cycle(rp_defeat, e):
                    rp_defeat.remove_edge(e[0], e[1])
        winners = unbeaten_candidates(rp_defeat)
    return sorted(list(set(winners)))

@vm_name("River")
def river(profile):
    """Order the edges in the weak margin graph from largest to smallest and lock them 
    in in that order, skipping edges that create a cycle.   Break ties using a tie-breaking 
    linear ordering over the edges.   A candidate is a Ranked Pairs winner if it wins according 
    to some tie-breaking rule. Also known as Tideman's Rule.
    """
    wmg = generate_weak_margin_graph(profile)
    cw = profile.condorcet_winner()
    # Ranked Pairs is Condorcet consistent, so simply return the Condorcet winner if exists
    if cw is not None: 
        winners = [cw]
    else:
        winners = list()            
        margins = sorted(list(set([e[2]['weight'] for e in wmg.edges(data=True)])), reverse=True)
        sorted_edges = [[e for e in wmg.edges(data=True) if e[2]['weight'] == m] for m in margins]
        
        if np.prod([math.factorial(len(es)) for es in sorted_edges]) > 1000: 
            #print("skipped", np.prod([math.factorial(len(es)) for es in sorted_edges]))
            return None
        else:
            #print("checked", np.prod([math.factorial(len(es)) for es in sorted_edges]))

            tbs = product(*[permutations(edges) for edges in sorted_edges])
            for tb in tbs:
                edges = flatten(tb)
                new_ranking = nx.DiGraph() 
                for e in edges: 
                    if e[1] not in new_ranking.nodes or len(list(new_ranking.in_edges(e[1]))) == 0:
                        new_ranking.add_edge(e[0], e[1], weight=e[2]['weight'])
                        if  does_create_cycle(new_ranking, e):
                            new_ranking.remove_edge(e[0], e[1])
                #print(new_ranking.edges)
                winners.append(unbeaten_candidates(new_ranking)[0])
    return sorted(list(set(winners)))

@vm_name("Ranked Pairs")
def ranked_pairs_with_test(profile):
    """Order the edges in the weak margin graph from largest to smallest and lock them 
    in in that order, skipping edges that create a cycle.   Break ties using a tie-breaking 
    linear ordering over the edges.   A candidate is a Ranked Pairs winner if it wins according 
    to some tie-breaking rule. Also known as Tideman's Rule.
    """
    wmg = generate_weak_margin_graph(profile)
    cw = profile.condorcet_winner()
    # Ranked Pairs is Condorcet consistent, so simply return the Condorcet winner if exists
    if cw is not None: 
        winners = [cw]
    else:
        winners = list()            
        margins = sorted(list(set([e[2]['weight'] for e in wmg.edges(data=True)])), reverse=True)
        sorted_edges = [[e for e in wmg.edges(data=True) if e[2]['weight'] == m] for m in margins]
        if np.prod([math.factorial(len(es)) for es in sorted_edges]) > 1000: 
            return None
        else: 
            tbs = product(*[permutations(edges) for edges in sorted_edges])
            for tb in tbs:
                edges = flatten(tb)
                new_ranking = nx.DiGraph() 
                for e in edges: 
                    new_ranking.add_edge(e[0], e[1], weight=e[2]['weight'])
                    if does_create_cycle(new_ranking, e):
                        new_ranking.remove_edge(e[0], e[1])
                winners.append(unbeaten_candidates(new_ranking)[0])
    return sorted(list(set(winners)))

@vm_name("Ranked Pairs")
def ranked_pairs_mg(mg):
    """Order the edges in the weak margin graph from largest to smallest and lock them 
    in in that order, skipping edges that create a cycle.   Break ties using a tie-breaking 
    linear ordering over the edges.   A candidate is a Ranked Pairs winner if it wins according 
    to some tie-breaking rule. Also known as Tideman's Rule.
    """
    
    wmg = mg_to_wmg(mg)
    cw = find_condorcet_winner(mg)
    # Ranked Pairs is Condorcet consistent, so simply return the Condorcet winner if exists
    
    if len(cw) == 1: 
        winners = cw
    else:
        winners = list()            
        margins = sorted(list(set([e[2]['weight'] for e in wmg.edges(data=True)])), reverse=True)
        sorted_edges = [[e for e in wmg.edges(data=True) if e[2]['weight'] == m] for m in margins]
        if np.prod([math.factorial(len(es)) for es in sorted_edges]) > 1000: 
            #print("skipped", np.prod([math.factorial(len(es)) for es in sorted_edges]))
            return None
        else:
            #print("checked", np.prod([math.factorial(len(es)) for es in sorted_edges]))

            tbs = product(*[permutations(edges) for edges in sorted_edges])
            
            for tb in tbs:
                edges = flatten(tb)
                new_ranking = nx.DiGraph() 
                for e in edges: 
                    new_ranking.add_edge(e[0], e[1], weight=e[2]['weight'])
                    if does_create_cycle(new_ranking, e):
                        new_ranking.remove_edge(e[0], e[1])
                winners.append(unbeaten_candidates(new_ranking)[0])
    return sorted(list(set(winners)))

@vm_name("River")
def river_mg(mg):
    """Order the edges in the weak margin graph from largest to smallest and lock them 
    in in that order, skipping edges that create a cycle.   Break ties using a tie-breaking 
    linear ordering over the edges.   A candidate is a Ranked Pairs winner if it wins according 
    to some tie-breaking rule. Also known as Tideman's Rule.
    """

    wmg = mg_to_wmg(mg)
    cw = find_condorcet_winner(mg)
    # Ranked Pairs is Condorcet consistent, so simply return the Condorcet winner if exists
    if len(cw) == 1: 
        winners = cw
    else:
        winners = list()            
        margins = sorted(list(set([e[2]['weight'] for e in wmg.edges(data=True)])), reverse=True)
        sorted_edges = [[e for e in wmg.edges(data=True) if e[2]['weight'] == m] for m in margins]
        
        if np.prod([math.factorial(len(es)) for es in sorted_edges]) > 1000: 
            #print("skipped", np.prod([math.factorial(len(es)) for es in sorted_edges]))
            return None
        else:
            #print("checked", np.prod([math.factorial(len(es)) for es in sorted_edges]))

            tbs = product(*[permutations(edges) for edges in sorted_edges])
            for tb in tbs:
                edges = flatten(tb)
                new_ranking = nx.DiGraph() 
                for e in edges: 
                    if e[1] not in new_ranking.nodes or len(list(new_ranking.in_edges(e[1]))) == 0:
                        new_ranking.add_edge(e[0], e[1], weight=e[2]['weight'])
                        if  does_create_cycle(new_ranking, e):
                            new_ranking.remove_edge(e[0], e[1])
                #print(new_ranking.edges)
                winners.append(unbeaten_candidates(new_ranking)[0])
    return sorted(list(set(winners)))



# ####
# Other methods
# ####

@vm_name("Iterated Removal Condorcet Loser")
def iterated_remove_cl(profile):
    """The winners are the candidates that survive iterated removal of 
    Condorcet losers
    """
    
    condorcet_loser = profile.condorcet_loser()  
    
    updated_profile = profile
    orig_cnames = {c:c for c in profile.candidates}
    while len(updated_profile.candidates) > 1 and  condorcet_loser is not None:    
        updated_profile, _orig_cnames = updated_profile.remove_candidates([condorcet_loser])
        orig_cnames = {c:orig_cnames[cn] for c,cn in _orig_cnames.items()}
        condorcet_loser = updated_profile.condorcet_loser()
            
    return sorted([orig_cnames[c] for c in updated_profile.candidates])


@vm_name("Daunou")
def daunou(profile):
    """Implementaiton of Daunou's voting method as described in the paper: 
    https://link.springer.com/article/10.1007/s00355-020-01276-w
    
    If there is a Condorcet winner, then that candidate is the winner.  Otherwise, 
    iteratively remove all Condorcet losers then select the plurality winner from among 
    the remaining conadidates
    """

    cw = profile.condorcet_winner()
    
    if cw is not None: 
        updated_profile = profile
        orig_cnames = {c:c for c in profile.candidates}
        winners = [cw]
    else: 
        cands_survive_it_rem_cl = iterated_remove_cl(profile)
        updated_profile, orig_cnames = profile.remove_candidates([_c for _c in profile.candidates 
                                                                  if _c not in cands_survive_it_rem_cl])
        winners = plurality(updated_profile)
        
    return sorted([orig_cnames[c] for c in winners])


@vm_name("Blacks")
def blacks(profile):
    """Blacks method returns the Condorcet winner if one exists, otherwise return the Borda winners.
    """
    
    cw = profile.condorcet_winner()
    
    if cw is not None:
        winners = [cw]
    else:
        winners = borda(profile)
        
    return winners


def simple_stable_voting_(profile, curr_cands = None, mem_sv_winners = {}): 
    '''
    Determine the Simple Stable Voting winners for the profile while keeping track 
    of the winners in any subprofiles checked during computation. 
    '''
    
    # curr_cands is the set of candidates who have not been removed
    curr_cands = curr_cands if not curr_cands is None else profile.candidates
    sv_winners = list()
    
    matches = [(a, b) for a in curr_cands for b in curr_cands if a != b]
    margins = list(set([profile.margin(a, b) for a,b in matches]))
        
    if len(curr_cands) == 1: 
        mem_sv_winners[tuple(curr_cands)] = curr_cands
        return curr_cands, mem_sv_winners
    for m in sorted(margins, reverse=True):
        for a, b in [ab_match for ab_match in matches 
                     if profile.margin(ab_match[0], ab_match[1])  == m]:
            if a not in sv_winners: 
                cands_minus_b = sorted([c for c in curr_cands if c!= b])
                if tuple(cands_minus_b) not in mem_sv_winners.keys(): 
                    ws, mem_sv_winners = simple_stable_voting_(profile, 
                                                               curr_cands = [c for c in curr_cands if c != b],
                                                               mem_sv_winners = mem_sv_winners)
                    mem_sv_winners[tuple(cands_minus_b)] = ws
                else: 
                    ws = mem_sv_winners[tuple(cands_minus_b)]
                if a in ws:
                    sv_winners.append(a)
        if len(sv_winners) > 0: 
            return sorted(sv_winners), mem_sv_winners
        
@vm_name("Simple Stable Voting")
def simple_stable_voting(profile): 
    '''Implementation of the Simple Stable Voting method from https://arxiv.org/abs/2108.00542'''
    return simple_stable_voting_(profile, curr_cands = None, mem_sv_winners = {})[0]


def find_strengths(profile, curr_cands = None):   
    """
    A path from candidate a to candidate b is a list of candidates starting with a and ending with b 
    such that each candidate in the list has a nonzero margin vs. the next candidate in the list. 
    The strength of a path is the minimum margin between consecutive candidates in the path 
    The strength of the pair of candidates (a,b) is strength of the strongest path from a to b.   
    We find these strengths using the Floyd-Warshall Algorithm.  
    """
    curr_cands = curr_cands if curr_cands is not None else profile.candidates
    mg = [[0 for _ in curr_cands] for _ in curr_cands]
    
    for c1_idx,c1 in enumerate(curr_cands):
        for c2_idx,c2 in enumerate(curr_cands):
            if (profile.support(c1,c2) > profile.support(c2,c1) or c1 == c2):
                mg[c1_idx][c2_idx] = profile.support(c1,c2) - profile.support(c2,c1)    
    strength = list(map(lambda i : list(map(lambda j : j , i)) , mg))
    for i_idx, i in enumerate(curr_cands):         
        for j_idx, j in enumerate(curr_cands): 
            if i!= j:
                for k_idx, k in enumerate(curr_cands): 
                    if i!= k and j != k:
                        strength[j_idx][k_idx] = max(strength[j_idx][k_idx], min(strength[j_idx][i_idx],strength[i_idx][k_idx]))
    return strength

def stable_voting_v1_(profile, curr_cands = None, mem_sv_winners = {}): 
    '''
    Determine the Stable Voting winners for the profile while keeping track 
    of the winners in any subprofiles checked during computation. 
    '''
    
    # curr_cands is the set of candidates who have not been removed
    curr_cands = curr_cands if not curr_cands is None else profile.candidates
    sv_winners = list()
    
    matches = [(a, b) for a in curr_cands for b in curr_cands if a != b]    
    margins = list(set([profile.margin(a, b) for a,b in matches]))
    nonneg_margins = [m for m in margins if m>=0]
    neg_margins = [m for m in margins if m < 0]
    
    if len(curr_cands) == 1: 
        mem_sv_winners[tuple(curr_cands)] = curr_cands
        return curr_cands, mem_sv_winners
    for m in sorted(nonneg_margins, reverse=True):
        for a, b in [ab_match for ab_match in matches 
                     if profile.margin(ab_match[0], ab_match[1])  == m]:
            if a not in sv_winners: 
                cands_minus_b = sorted([c for c in curr_cands if c!= b])
                if tuple(cands_minus_b) not in mem_sv_winners.keys(): 
                    ws, mem_sv_winners = stable_voting_v1_(profile, 
                                                           curr_cands = [c for c in curr_cands if c != b],
                                                           mem_sv_winners = mem_sv_winners)
                    mem_sv_winners[tuple(cands_minus_b)] = ws
                else: 
                    ws = mem_sv_winners[tuple(cands_minus_b)]
                if a in ws:
                    sv_winners.append(a)
        if len(sv_winners) > 0: 
            return sorted(sv_winners), mem_sv_winners
    
    strengths = find_strengths(profile, curr_cands)
    for m in sorted(neg_margins, reverse=True):
        for a, b in [ab_match for ab_match in matches 
                     if profile.margin(ab_match[0], ab_match[1])  == m]:
            if strengths[curr_cands.index(a)][curr_cands.index(b)] >= profile.margin(b, a) and a not in sv_winners: 
                cands_minus_b = sorted([c for c in curr_cands if c!= b])
                if tuple(cands_minus_b) not in mem_sv_winners.keys(): 
                    ws, mem_sv_winners = stable_voting_v1_(profile, 
                                                           curr_cands = [c for c in curr_cands if c != b],
                                                           mem_sv_winners = mem_sv_winners)
                    mem_sv_winners[tuple(cands_minus_b)] = ws
                else: 
                    ws = mem_sv_winners[tuple(cands_minus_b)]
                if a in ws:
                    sv_winners.append(a)
        if len(sv_winners) > 0: 
            return sorted(sv_winners), mem_sv_winners
                
@vm_name("Stable Voting (V1)")
def stable_voting_v1(profile): 
    return stable_voting_v1_(profile, curr_cands = None, mem_sv_winners = {})[0]


def find_undefeated_candidates(profile, curr_cands = None):   
    """
    A path from candidate a to candidate b is a list of candidates starting with a and ending with b 
    such that each candidate in the list has a nonzero margin vs. the next candidate in the list. 
    The strength of a path is the minimum margin between consecutive candidates in the path 
    The strength of the pair of candidates (a,b) is strength of the strongest path from a to b.   
    We find these strengths using the Floyd-Warshall Algorithm.   
    
    A candidate c is undefeated if for all candidates d that is majority preferred to d, 
    the margin of c over d is greater than the strength of d to c. 
    
    Note that this is an implementation of the Split Cycle voting method. 
    """
    curr_cands = curr_cands if curr_cands is not None else profile.candidates
    mg = [[-np.inf for _ in curr_cands] for _ in curr_cands]
    
    for c1_idx,c1 in enumerate(curr_cands):
        for c2_idx,c2 in enumerate(curr_cands):
            if (profile.support(c1,c2) > profile.support(c2,c1) or c1 == c2):
                mg[c1_idx][c2_idx] = profile.support(c1,c2) - profile.support(c2,c1)    
    strength = list(map(lambda i : list(map(lambda j : j , i)) , mg))
    for i_idx, i in enumerate(curr_cands):         
        for j_idx, j in enumerate(curr_cands): 
            if i!= j:
                for k_idx, k in enumerate(curr_cands): 
                    if i!= k and j != k:
                        strength[j_idx][k_idx] = max(strength[j_idx][k_idx], min(strength[j_idx][i_idx],strength[i_idx][k_idx]))
    undefeated = {i:True for i in curr_cands}
    for i_idx, i in enumerate(curr_cands): 
        for j_idx,j in enumerate(curr_cands):
            if i!=j:
                if mg[j_idx][i_idx] > strength[i_idx][j_idx]: # the main difference with Beat Path
                    undefeated[i] = False
    return sorted([c for c in curr_cands if undefeated[c]])


def stable_voting_(profile, curr_cands = None, mem_sv_winners = {}): 
    '''
    Determine the Stable Voting winners for the profile while keeping track 
    of the winners in any subprofiles checked during computation. 
    '''
    
    # curr_cands is the set of candidates who have not been removed
    curr_cands = curr_cands if not curr_cands is None else profile.candidates
    sv_winners = list()
    
    undefeated_candidates = find_undefeated_candidates(profile, curr_cands = curr_cands)
    matches = [(a, b) for a in curr_cands for b in curr_cands if a != b if a in undefeated_candidates]
    margins = list(set([profile.margin(a, b) for a,b in matches]))
        
    if len(undefeated_candidates) == 1: 
        mem_sv_winners[tuple(curr_cands)] = curr_cands
        return curr_cands, mem_sv_winners
    for m in sorted(margins, reverse=True):
        for a, b in [ab_match for ab_match in matches 
                     if profile.margin(ab_match[0], ab_match[1])  == m]:
            if a not in sv_winners: 
                cands_minus_b = sorted([c for c in curr_cands if c!= b])
                if tuple(cands_minus_b) not in mem_sv_winners.keys(): 
                    ws, mem_sv_winners = stable_voting_(profile, 
                                                        curr_cands = [c for c in curr_cands if c != b],
                                                        mem_sv_winners = mem_sv_winners)
                    mem_sv_winners[tuple(cands_minus_b)] = ws
                else: 
                    ws = mem_sv_winners[tuple(cands_minus_b)]
                if a in ws:
                    sv_winners.append(a)
        if len(sv_winners) > 0: 
            return sorted(sv_winners), mem_sv_winners
        
@vm_name("Stable Voting")
def stable_voting(profile): 
    return stable_voting_(profile, curr_cands = None, mem_sv_winners = {})[0]

@vm_name("Stable Voting Faster")
def stable_voting_faster(profile): 
    '''First check if there is a Condorcet winner.  If so, return the Condorcet winner, otherwise 
    find the stable voting winnner using stable_voting_'''
    
    cw = profile.condorcet_winner()
    if cw is not None: 
        return [cw]
    else: 
        return stable_voting_(profile, curr_cands = None, mem_sv_winners = {})[0]


def display_mg_with_sc(profile, cmap = None): 
    '''Display a margin graph for a profile where the SC defeat edges are highlighted blue and 
    the Split Cycle winners are highlighted blue, and others candidates are highlighted red.'''
    
    cmap = cmap if cmap is not None else profile.cmap

    mg = profile.margin_graph()
    sc_defeat = split_cycle_defeat(profile)
    sc_winners =  unbeaten_candidates(sc_defeat)
    
    label_dict = dict()
    for n in mg.nodes: 
        label_dict[n] = cmap[n]

    edges = mg.edges()
    sc_edges = sc_defeat.edges()
    
    colors = ['blue' if e in sc_edges else 'black' for e in edges]
    widths = [3 if e in sc_edges else 1.5 for e in edges]
    
    pos = nx.circular_layout(mg)
    nx.draw(mg, pos, edges=edges, edge_color=colors, width=widths,
            font_size=20, node_color=['blue' if n in sc_winners else 'red' for n in mg.nodes], font_color='white', node_size=700, 
            labels=label_dict, with_labels=True)
    labels = nx.get_edge_attributes(mg,'weight')
    nx.draw_networkx_edge_labels(mg,pos,edge_labels=labels, font_size=14, label_pos=0.3)
    plt.show()


### Functions for reasoning about margin graphs
def is_uniquely_weighted(mg): 
    
    return all([(mg.has_edge(c1,c2) or mg.has_edge(c2,c1)) for c1 in mg.nodes for c2 in mg.nodes if c1 != c2]) and len(list(set([e[2] for e in mg.edges.data("weight") ]))) == len(mg.edges)

def display_graph(g): 
    pos = nx.circular_layout(g)
    nx.draw(g, pos,  width=1.5,
            font_size=20, node_color='blue', font_color='white', node_size=700, 
            with_labels=True)
    #labels = nx.get_edge_attributes(mg,'weight')
    #nx.draw_networkx_edge_labels(mg,pos, font_size=14, label_pos=0.3)
    plt.show()

def display_mg_with_sc_mg(mg): 
    
    sc_defeat = split_cycle_defeat_mg(mg)
    sc_winners =  unbeaten_candidates(sc_defeat)
    edges = mg.edges()
    sc_edges = sc_defeat.edges()
    
    colors = ['blue' if e in sc_edges else 'black' for e in edges]
    widths = [3 if e in sc_edges else 1.5 for e in edges]
    
    pos = nx.circular_layout(mg)
    nx.draw(mg, pos, edges=edges, edge_color=colors, width=widths,
            font_size=20, node_color=['blue' if n in sc_winners else 'red' for n in mg.nodes], font_color='white', node_size=700, 
            with_labels=True)
    labels = nx.get_edge_attributes(mg,'weight')
    nx.draw_networkx_edge_labels(mg,pos,edge_labels=labels, font_size=14, label_pos=0.3)
    plt.show()

def minimax_scores_mg(mg): 
    
    return {c: -max([mg[in_e[0]][in_e[1]]['weight']  for in_e in mg.in_edges(c)]) if len(mg.in_edges(c)) > 0 else 0 for c in mg.nodes}


def get_margin(mg, a, b): 
    '''get the margin of a over b in the marging graph mg'''
    m = 0.0
    
    if mg.has_edge(a, b): 
        m = mg.get_edge_data(a, b)['weight'] 
    elif mg.has_edge(b, a):
        m = -1 * mg.get_edge_data(b, a)['weight'] 
    return m

def simple_stable_voting_mg_(mg, curr_cands = None, mem_sv_winners = {}): 
    '''
    Determine the Simple Stable Voting winners for the margin graph mg while keeping track 
    of the winners in any subprofiles checked during computation. 
    '''
    
    # curr_cands is the set of candidates who have not been removed
    curr_cands = curr_cands if not curr_cands is None else mg.nodes 
    sv_winners = list()

    matches = [(a, b) for a in curr_cands for b in curr_cands if a != b]
    margins = list(set([get_margin(mg, a, b) for a,b in matches]))
    
    if len(curr_cands) == 1: 
        mem_sv_winners[tuple(curr_cands)] = curr_cands
        return curr_cands, mem_sv_winners
    for m in sorted(margins, reverse=True):
        for a, b in [ab_match for ab_match in matches 
                     if get_margin(mg, ab_match[0], ab_match[1]) == m]:
            if a not in sv_winners: 
                cands_minus_b = sorted([c for c in curr_cands if c!= b])
                if tuple(cands_minus_b) not in mem_sv_winners.keys(): 
                    ws, mem_sv_winners = simple_stable_voting_mg_(mg, 
                                                                  curr_cands = [c for c in curr_cands if c != b],
                                                                  mem_sv_winners = mem_sv_winners )
                    mem_sv_winners[tuple(cands_minus_b)] = ws
                else: 
                    ws = mem_sv_winners[tuple(cands_minus_b)]
                if a in ws:
                    sv_winners.append(a)
        if len(sv_winners) > 0: 
            return sorted(sv_winners), mem_sv_winners
                
@vm_name("Simple Stable Voting")
def simple_stable_voting_mg(mg): 
    
    return simple_stable_voting_mg_(mg, curr_cands = None, mem_sv_winners = {})[0]

def find_strengths_mg(mg, curr_cands = None):   
    """
    A path from candidate a to candidate b is a list of candidates  starting with a and ending with b 
    such that each candidate in the list beats the next candidate in the list. 
    The strength of a path is the minimum margin between consecutive candidates in the path 
    The strength of the pair of candidates (a,b) is strength of the strongest path from a to b.   
    We find these strengths using the Floyd-Warshall Algorithm.  

    """
    curr_cands = curr_cands if curr_cands is not None else mg.nodes
    margin_matrix = [[0 for _ in curr_cands] for _ in curr_cands]
    
    # Weak Condorcet winners are Split Cycle winners
    for c1_idx,c1 in enumerate(curr_cands):
        for c2_idx,c2 in enumerate(curr_cands):
            if get_margin(mg, c1, c2) > 0 or c1 == c2:
                margin_matrix[c1_idx][c2_idx] = get_margin(mg, c1, c2)

    strength = list(map(lambda i : list(map(lambda j : j , i)) , margin_matrix))
    for i_idx, i in enumerate(curr_cands):         
        for j_idx, j in enumerate(curr_cands): 
            if i!= j:
                for k_idx, k in enumerate(curr_cands): 
                    if i!= k and j != k:
                        strength[j_idx][k_idx] = max(strength[j_idx][k_idx], min(strength[j_idx][i_idx],strength[i_idx][k_idx]))
    return strength

def stable_voting_v1_mg_(mg, curr_cands = None, mem_sv_winners = {}): 
    '''
    Determine the Simple Stable Voting winners for the margin graph mg while keeping track 
    of the winners in any subprofiles checked during computation. 
    '''
    
    # curr_cands is the set of candidates who have not been removed
    curr_cands = curr_cands if not curr_cands is None else list(mg.nodes)
    sv_winners = list()

    matches = [(a, b) for a in curr_cands for b in curr_cands if a != b]
    margins = list(set([get_margin(mg, a, b) for a,b in matches]))
    nonneg_margins = [m for m in margins if m >= 0]
    neg_margins = [m for m in margins if m < 0]
    if len(curr_cands) == 1: 
        mem_sv_winners[tuple(curr_cands)] = curr_cands
        return curr_cands, mem_sv_winners
    for m in sorted(nonneg_margins, reverse=True):
        for a, b in [ab_match for ab_match in matches 
                     if get_margin(mg, ab_match[0], ab_match[1]) == m]:
            if a not in sv_winners: 
                cands_minus_b = sorted([c for c in curr_cands if c!= b])
                if tuple(cands_minus_b) not in mem_sv_winners.keys(): 
                    ws, mem_sv_winners = stable_voting_v1_mg_(mg, 
                                                           curr_cands = [c for c in curr_cands if c != b],
                                                           mem_sv_winners = mem_sv_winners )
                    mem_sv_winners[tuple(cands_minus_b)] = ws
                else: 
                    ws = mem_sv_winners[tuple(cands_minus_b)]
                if a in ws:
                    sv_winners.append(a)
        if len(sv_winners) > 0: 
            return sorted(sv_winners), mem_sv_winners
        
    strengths = find_strengths_mg(mg, curr_cands)
    for m in sorted(neg_margins, reverse=True): 
        
        for a, b in [ab_match for ab_match in matches 
                     if get_margin(mg, ab_match[0], ab_match[1]) == m]:
            if strengths[curr_cands.index(a)][curr_cands.index(b)] >= get_margin(mg, b, a) and a not in sv_winners: 
                cands_minus_b = sorted([c for c in curr_cands if c!= b])
                if tuple(cands_minus_b) not in mem_sv_winners.keys(): 
                    ws, mem_sv_winners = stable_voting_v1_mg_(mg, 
                                                           curr_cands = [c for c in curr_cands if c != b],
                                                           mem_sv_winners = mem_sv_winners )
                    mem_sv_winners[tuple(cands_minus_b)] = ws
                else: 
                    ws = mem_sv_winners[tuple(cands_minus_b)]
                if a in ws:
                    sv_winners.append(a)
        if len(sv_winners) > 0: 
            return sorted(sv_winners), mem_sv_winners
            
@vm_name("Stable Voting (V1)")
def stable_voting_v1_mg(mg): 
    
    return stable_voting_v1_mg_(mg, curr_cands = None, mem_sv_winners = {})[0]

def find_undefeated_candidates_mg(mg, curr_cands = None):   
    """
    A path from candidate a to candidate b is a list of candidates starting with a and ending with b 
    such that each candidate in the list has a nonzero margin vs. the next candidate in the list. 
    The strength of a path is the minimum margin between consecutive candidates in the path 
    The strength of the pair of candidates (a,b) is strength of the strongest path from a to b.   
    We find these strengths using the Floyd-Warshall Algorithm.   
    
    A candidate c is undefeated if for all candidates d that is majority preferred to d, 
    the margin of c over d is greater than the strength of d to c. 
    
    Note that this is an implementation of the Split Cycle voting method. 
    """
    curr_cands = curr_cands if curr_cands is not None else mg.nodes
    margin_matrix = [[-np.inf for _ in curr_cands] for _ in curr_cands]
    
    for c1_idx,c1 in enumerate(curr_cands):
        for c2_idx,c2 in enumerate(curr_cands):
            if get_margin(mg, c1, c2) > 0 or c1 == c2:
                margin_matrix[c1_idx][c2_idx] = get_margin(mg, c1, c2)

    strength = list(map(lambda i : list(map(lambda j : j , i)) , margin_matrix))
    for i_idx, i in enumerate(curr_cands):         
        for j_idx, j in enumerate(curr_cands): 
            if i!= j:
                for k_idx, k in enumerate(curr_cands): 
                    if i!= k and j != k:
                        strength[j_idx][k_idx] = max(strength[j_idx][k_idx], min(strength[j_idx][i_idx],strength[i_idx][k_idx]))

    undefeated = {i:True for i in curr_cands}
    for i_idx, i in enumerate(curr_cands): 
        for j_idx,j in enumerate(curr_cands):
            if i!=j:
                if margin_matrix[j_idx][i_idx] > strength[i_idx][j_idx]: # the main difference with Beat Path
                    undefeated[i] = False
    return sorted([c for c in curr_cands if undefeated[c]])

def stable_voting_mg_(mg, curr_cands = None, mem_sv_winners = {}): 
    '''
    Determine the Simple Stable Voting winners for the margin graph mg while keeping track 
    of the winners in any subprofiles checked during computation. 
    '''
    
    # curr_cands is the set of candidates who have not been removed
    curr_cands = curr_cands if not curr_cands is None else mg.nodes 
    sv_winners = list()

    undefeated_candidates = find_undefeated_candidates_mg(mg, curr_cands = curr_cands)
    matches = [(a, b) for a in curr_cands for b in curr_cands if a != b if a in undefeated_candidates]
    margins = list(set([get_margin(mg, a, b) for a,b in matches]))
    
    if len(undefeated_candidates) == 1: 
        mem_sv_winners[tuple(curr_cands)] = curr_cands
        return curr_cands, mem_sv_winners
    for m in sorted(margins, reverse=True):
        for a, b in [ab_match for ab_match in matches 
                     if get_margin(mg, ab_match[0], ab_match[1]) == m]:
            if a not in sv_winners: 
                cands_minus_b = sorted([c for c in curr_cands if c!= b])
                if tuple(cands_minus_b) not in mem_sv_winners.keys(): 
                    ws, mem_sv_winners = stable_voting_mg_(mg, 
                                                           curr_cands = [c for c in curr_cands if c != b],
                                                           mem_sv_winners = mem_sv_winners )
                    mem_sv_winners[tuple(cands_minus_b)] = ws
                else: 
                    ws = mem_sv_winners[tuple(cands_minus_b)]
                if a in ws:
                    sv_winners.append(a)
        if len(sv_winners) > 0: 
            return sorted(sv_winners), mem_sv_winners
                
@vm_name("Stable Voting")
def stable_voting_mg(mg): 
    
    return stable_voting_mg_(mg, curr_cands = None, mem_sv_winners = {})[0]

def display_winners(profile, vm): 
    ws = vm(profile)
    winners = ', '.join([profile.cmap[w] for w in ws])
    print(f"{vm.name} winner{'s' if len(ws) > 1 else ''}: {winners}")

all_vms = [
    plurality,
    borda, 
    anti_plurality,
    hare,
    hare_tb, 
    hare_put,
    plurality_with_runoff,
    coombs,
    coombs_tb,
    coombs_put,
    baldwin,
    baldwin_tb,
    baldwin_put,
    strict_nanson, 
    weak_nanson,
    bucklin,
    simplified_bucklin,
    condorcet,
    copeland,
    llull,
    uc_gill,
    uc_fish,
    uc_bordes,
    uc_mckelvey,
    getcha,
    gocha,
    minimax, 
    split_cycle,
    split_cycle_faster,
    beat_path,
    beat_path_faster,
    ranked_pairs,
    ranked_pairs_zt,
    ranked_pairs_t,
    iterated_remove_cl,
    stable_voting,
    simple_stable_voting,
    stable_voting_faster,
    daunou,
    blacks
]

all_vms_mg = [
    condorcet_mg,
    copeland_mg,
    llull_mg,
    uc_gill_mg,
    uc_fish_mg,
    uc_bordes_mg, 
    uc_mckelvey,
    getcha_mg,
    gocha_mg,
    stable_voting_mg,
    simple_stable_voting_mg,
    split_cycle_mg,
    beat_path_faster_mg,
    split_cycle_faster_mg
]

    

