
import itertools
from prettytable import *
from math import ceil
###
#Helper functions
###

def generate_linear_orderings(candidates): 
    # returns all linear orderings for a set of candidates
    return list(itertools.permutations(candidates))

def create_subsets(s, n): 
    # return all subsets of s of size n
    return list(itertools.combinations(s, n)) 

## compiled functions
import numpy as np
import numba
from numba import jit

@jit(nopython=True)
def _support(prof, c1, c2):
    
    _diffs = prof[0:,c1] - prof[0:,c2]
    _diffs[_diffs > 0] = 0
    _diffs[_diffs < 0] = 1
    return np.sum(_diffs)

@jit(nopython=True)
def _margin(prof, c1, c2): 
    return _support(prof,  c1, c2) - _support(prof, c2, c1)


@jit(nopython=True)
def _num_rank(prof_cands, cand, level):
    cands_at_level =  prof_cands[0:,level-1]

    is_cand = cands_at_level  == cand
    
    return cands_at_level[is_cand].size

@jit(nopython=True)
def _borda_score(prof_cands, num_cands, cand):
    
    bscores = np.arange(num_cands)[::-1]
    levels = np.arange(1,num_cands+1)
    
    num_ranks = np.array([_num_rank(prof_cands, cand, level) for level in levels])
    return  np.sum(num_ranks * bscores)

# Profiles
class ProfileOpt(object):
    
    def __init__(self, _prof, cand_names, voter_names, cmap=None):
        
        self.cand_names = list(cand_names)

        self.cmap = cmap if cmap is not None else {c:c for c in self.cand_names}
        
        self.cand_list = range(0, len(cand_names))
        
        self.voter_names = list(voter_names)
        
        self._ranks = np.array([[_p.index(c) + 1 for c in self.cand_names] for _p in _prof])
        self._cands = np.array([[self.cand_names.index(c) for c in _p] for _p in _prof])
    
    #@property
    #def voters(self):
    #    return self._profile
    
    @property 
    def candidates(self):
        return self.cand_list
    
    @property 
    def cand_pairs(self):
        return itertools.combinations(self.candidates, 2)
    
    @property
    def num_candidates(self):
        return len(self.candidates)
    
    @property 
    def size(self):
        return len(self.voter_names), self.num_candidates
    
    @property
    def all_rankings(self):
        return generate_linear_orderings(self.candidates)
    
    def cname(self, c):
        return self.cmap[self.cand_names[c]]
    
    def support(self, c1, c2):
        # wrapper to call the compiled _support function
        
        return _support(self._ranks, c1, c2)
    
    def margin(self, c1, c2):

        # wrapper to call the compiled _margin function
        return _margin(self._ranks, c1, c2)

    def create_new_profile(self, voter, new_ordering):
        # given a voter and a new ordering, create a new profile exactly like the existing one
        
        new_prof = [list(p) if vidx == self.voter_names.index(voter) else new_ordering 
                    for vidx, p in enumerate(self._profile)]
                
        return ProfileOpt(new_prof, self.cand_names, self_voter_names)

    def remove_candidates(self, cands_to_remove):
        # create a new profile with the candidates removed
        
        new_prof = [[self.cand_names[c] for c in p if c not in cands_to_remove] for p in  self._cands]

        new_cand_names = [self.cand_names[c] for c in self.candidates if c not in cands_to_remove]
        
        return ProfileOpt(new_prof, new_cand_names, self.voter_names, self.cmap)
    
    def remove_candidates_by_name(self, cands_by_name_to_remove):
        # create a new profile with the candidates removed
        
        new_prof = [[self.cand_names[c] for c in p if self.cand_names[c] not in cands_by_name_to_remove] for p in  self._cands]

        new_cand_names = [self.cand_names[c] for c in self.candidates if self.cand_names[c] not in cands_by_name_to_remove]
        
        return ProfileOpt(new_prof, new_cand_names, self.voter_names, self.cmap)

    def is_empty(self):
        # return true if some ranking is empty
        return any([len(p) == 0 for p in self._cands])
    
    def tally(self):

        # generate a tally  for all the candidates.   This is a dictionary associating with each voter, the 
        # margin of victory for that candidate over the other candidates. 

        return {c1: {c2: self.margin(c1,c2) for c2 in self.candidates if c2 != c1} 
                for c1 in self.candidates}

    def majority_prefers(self, c1, c2): 
        # return True if more voters rank c1 over c2 than c2 over c1

        return self.margin(c1,c2) > 0

    def num_rank(self, cand, level = 1):
        # returns the number of voters that rank cand at level
        
        return _num_rank(self._cands, cand, level)

    def plurality_scores(self):
        #  return  the pluarlity tally
        
        return {c: self.num_rank(c, level=1) for c in self.candidates}

    def borda_scores(self):
        # return the Borda scores for each candidate
        
        borda_scores = {c:0 for c in self.candidates}
        num_candidates = len(self.candidates)
        borda_scores = {c: _borda_score(self._cands, num_candidates, c) for c in self.candidates}
        return borda_scores
    
    def strict_maj_size(self):
        
        # return the size of  > 50% of the voters
        return len(self.voter_names)/2 + 1 if len(self.voter_names) % 2 == 0 else int(ceil(float(len(self.voter_names))/2))

    def cand_list_str(self, clist):
        return "{" +  ", ".join([self.cand_names[c] for c in clist]) + "}"

    def add_voter(self, ranking):
        '''create a new profile with an additional voter with the given ranking'''
        
        new_prof = [tuple(r) for r in self._cands] + [ranking]
        new_voters = self.voter_names + ['new']
        
        return ProfileOpt(new_prof, self.cand_names, new_voters, self.cmap)
        

    def display(self, cmap=None):
        tbl = PrettyTable()
        
        for v,r in enumerate(self._cands):
            tbl.add_column(str(self.voter_names[v]), [cmap[self.cand_names[c]] if cmap != None else self.cand_names[c] for c in r])
        print tbl

    def display_anon(self):
        tbl = PrettyTable()
        
        prof_count = {r:0 for r in list(set(list([tuple(_) for _ in self._cands])))}

        for _r in self._cands:
            prof_count[tuple(_r)] += 1
     
        for r in prof_count.keys():
            tbl.add_column(str(prof_count[r]), [self.cmap[self.cand_names[c]] for c in r])
        print tbl

    def get_profile_string(self):
        tbl = PrettyTable()
        for  v,r in enumerate(self._cands):
            tbl.add_column(str(v), [self.cmap[self.cand_names[c]] for c in r])
        return str(tbl)

    def as_dict(self): 
        return {str(v): tuple(r) for v,r in enumerate(self._cands)}
    
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.voter_names == other.voter_names and self.cand_names == other.cand_names and all([np.array_equal(ranks[0], ranks[1]) for ranks in zip(self._cands,other._cands)])
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)
    

###
#Helper functions
###

def create_profile_opt(rankings, voter_names=None, candidate_names=None, cmap=None):
    # given voter names and a list of rankings, return a profile
    
    assert len(voter_names) == len(rankings), "Error: the number of voters doesn't match the number of rankings"
    
    voter_names = voter_names if voter_names != None else list(range(len(rankings)))
    candidate_names = candidate_names if candidate_names != None else list(range(len(rankings[0])))
    
    return ProfileOpt(rankings, candidate_names, voter_names, cmap)

def create_profile_opt_from_anon_profile(anon_rankings, candidate_names=None, cmap=None): 
    
    candidate_names = candidate_names if candidate_names != None else list(range(len(rankings[0])))
    rankings = list()
    voter_names = list()
    voter_idx = 0
    for _prof in anon_rankings.keys():
        for i in range(0,anon_rankings[_prof]):
            voter_names.append('v' + str(voter_idx+i))
            rankings.append(_prof)
        voter_idx += anon_rankings[_prof]
    return create_profile_opt(rankings, voter_names, candidate_names, cmap)
    
