# Tools to generate profiles

from preflibtools.generate_profiles import *
import string
from itertools import product, combinations
import copy
def generate_names_opt(num_cands, num_voters):

    cand_names = range(0,num_cands)

    voter_names = basic_voter_names = list(string.ascii_lowercase)

    while len(voter_names) < num_voters: 
        voter_names = voter_names + [''.join(_c) for _c in  list(product(voter_names, basic_voter_names))]
    
    return cand_names, voter_names[0:num_voters]


def create_prof_opt(num_voters, cmap):
    num_cands = len(cmap.keys())
    rmaps, rmapscounts = gen_impartial_culture_strict(num_voters, cmap)

    _prof = list()
    for _rmap in zip(rmapscounts,rmaps):

        _rank = list()
        for r in range(1,num_cands + 1):
            _rank.append(cmap[_rmap[1].keys()[_rmap[1].values().index(r)]])
        _prof += [tuple(_rank)] *_rmap[0]
        
    return _prof

def get_random_ranking(cmap):
    refm, refc = gen_impartial_culture_strict(1, cmap)
    
    return io.rankmap_to_order(refm[0])

def create_prof_opt_mallows(num_voters, cmap, phi, ref=None):
    '''create a profile using a mallows model with dispersion param phi
    ref is the reference ranking (using preflib cand names!).   
    Generate at random'''
    
    if ref == None:
        ref = get_random_ranking(cmap)
    num_cands = len(cmap.keys())
    
    rmaps, rmapscounts = gen_mallows(num_voters, cmap, [1.0], [phi], [ref])

    _prof = list()
    for _rmap in zip(rmapscounts,rmaps):

        _rank = list()
        for r in range(1,num_cands + 1):
            _rank.append(cmap[_rmap[1].keys()[_rmap[1].values().index(r)]])
        _prof += [tuple(_rank)] *_rmap[0]
        
    return _prof

def create_prof_opt_mallows_two_rankings(num_voters, cmap, phi, ref=None):
    '''create a profile using a mallows model with dispersion param phi
    ref is the reference ranking (using preflib cand names!).   
    Generate at random'''
    
    if ref == None:
        ref = get_random_ranking(cmap)
        
    ref1 = copy.deepcopy(ref)
    ref.reverse()
    ref2 = copy.deepcopy(ref)
    num_cands = len(cmap.keys())
    
    rmaps, rmapscounts = gen_mallows(num_voters, cmap, [0.5, 0.5], [phi, phi], [ref1, ref2])

    _prof = list()
    for _rmap in zip(rmapscounts,rmaps):

        _rank = list()
        for r in range(1,num_cands + 1):
            _rank.append(cmap[_rmap[1].keys()[_rmap[1].values().index(r)]])
        _prof += [tuple(_rank)] *_rmap[0]
        
    return _prof


def generate_candidate_maps(cand_names):
    
    num_cands = len(cand_names)
    cmap =  gen_cand_map(num_cands)
    for i,c in enumerate(cand_names):
        cmap[i+1] = c
    return cmap