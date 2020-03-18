
from math import ceil
import numpy as np
import random
from itertools import permutations, product
import networkx as nx
import itertools
import copy
from voting.profile_optimized import generate_linear_orderings

# Voting methods

## helper functions

def generate_margin_graph(prof, min_margin = 0):
    '''generate the weighted margin graph'''
    
    mg = nx.DiGraph()
    mg.add_nodes_from(prof.candidates)
    mg.add_weighted_edges_from([(c1,c2,prof.support(c1,c2) - prof.support(c2,c1))  
           for c1 in prof.candidates 
           for c2 in prof.candidates if c1 != c2 if prof.support(c1,c2) - prof.support(c2,c1) > min_margin])
    return mg
    
def generate_weak_margin_graph(prof, min_margin = 0):
    '''generate the weak weighted margin graph'''
    
    mg = nx.DiGraph()
    mg.add_nodes_from(prof.candidates)
    mg.add_weighted_edges_from([(c1,c2,prof.support(c1,c2) - prof.support(c2,c1))  
           for c1 in prof.candidates 
           for c2 in prof.candidates if c1 != c2 if prof.support(c1,c2) - prof.support(c2,c1) >= min_margin])
    return mg

def has_cycle(G):
    try:
        cycles =  nx.find_cycle(G)
    except:
        cycles = list()
    return len(cycles) != 0

def unbeaten_candidates(G): 
    
    return [n for n in G.nodes if G.in_degree(n) == 0]

def find_condorcet_winner(G): 
    
    return [n for n in G.nodes if G.out_degree(n) == len(G.nodes) -  1]

flatten = lambda l: [item for sublist in l for item in sublist]


# Splity Cycle implementation 
def split_cycle(prof):
    """Split Cycle"""
    
    # create margin graph
    mg = generate_margin_graph(prof) 

    # find cycle numbers for each candidate
    cycle_number = {cs:0 for cs in combinations(prof.candidates,2)}
    for cycle in nx.simple_cycles(mg): # for each cycle in mg
        
        # get all the margins (i.e., the weights) of the edges in the cycle
        margins = list() 
        for idx,c1 in enumerate(cycle): 
            next_idx = idx + 1 if (idx + 1) < len(cycle) else 0
            c2 = cycle[next_idx]
            margins.append(mg[c1][c2]['weight'])
            
        min_margin = min(margins) # the min is the minimal margin needed to break the cycle
        
        # check if should update the cycle number for each candidate
        for cands in cycle_number.keys():
            if cands[0] in cycle and cands[1] in cycle:
                cycle_number[cands] = min_margin if min_margin > cycle_number[cands] else cycle_number[cands]
    
    # generate the beaten graph, where A beats B if margin(A,B) > cycle(A,B)
    
    cycle =  lambda c1,c2: cycle_number[[cs for cs in cycle_number.keys() if set(cs) == set([c1,c2])][0]]
    G = nx.DiGraph()
    G.add_nodes_from(prof.candidates)
    G.add_edges_from([(c1,c2)  
           for c1 in prof.candidates 
           for c2 in prof.candidates if c1 != c2 if prof.margin(c1,c2)  > cycle(c1,c2)])

    winners = unbeaten_candidates(G)
    return sorted(list(set(winners)))

# Return cycle numbers and split numbers
def split_cycle_with_data(prof):
    """Split Cycle"""
    
    # create margin graph
    mg = generate_margin_graph(prof) 

    # find cycle numbers for each candidate
    cycle_number = {cs:0 for cs in combinations(prof.candidates,2)}
    split_number = dict()
    for cycle in nx.simple_cycles(mg): # for each cycle in mg
        
        # get all the margins (i.e., the weights) of the edges in the cycle
        margins = list() 
        for idx,c1 in enumerate(cycle): 
            next_idx = idx + 1 if (idx + 1) < len(cycle) else 0
            c2 = cycle[next_idx]
            margins.append(mg[c1][c2]['weight'])
            
        min_margin = min(margins) # the min is the minimal margin needed to break the cycle
        
        split_number.update({tuple(cycle): min_margin})
        # check if should update the cycle number for each candidate
        for cands in cycle_number.keys():
            if cands[0] in cycle and cands[1] in cycle:
                cycle_number[cands] = min_margin if min_margin > cycle_number[cands] else cycle_number[cands]
    
    # generate the beaten graph, where A beats B if margin(A,B) > cycle(A,B)
    
    cycle =  lambda c1,c2: cycle_number[[cs for cs in cycle_number.keys() if set(cs) == set([c1,c2])][0]]
    G = nx.DiGraph()
    G.add_nodes_from(prof.candidates)
    G.add_edges_from([(c1,c2)  
           for c1 in prof.candidates 
           for c2 in prof.candidates if c1 != c2 if prof.margin(c1,c2)  > cycle(c1,c2)])

    winners = unbeaten_candidates(G)
    return sorted(list(set(winners))), cycle_number, split_number

# Display all the Split Cycle data
def display_split_cycle_data(prof, cmap):
    
    sc_winners, cycle_number, split_number = split_cycle_with_data(prof)

    display_winners(split_cycle, prof)
    print "\n---\n"

    for cands in cycle_number.keys():
        print "CycleNum({}, {}) = {}".format(cmap[cands[0]],cmap[cands[1]], cycle_number[cands])

    print "\n---\n"
    for cycle in split_number.keys():
        print "SplitNum({}) = {}".format(",".join([cmap[_] for _ in cycle]), split_number[cycle])
    
    print "\n"



def schulze_beatpath(prof): 
    """Beat Path"""
    
    #1. calculate vote_graph, edge from c1 to c2 of c1 beats c2, weighted by support for c1 over c2
    #2. For all pairs c1, c2, find all paths from c1 to c2, for each path find the minimum weight.  
    #   beatpath[c1,c2] = max(weight(p) all p's from c1 to c2)
    #3. winner is the candidates that beat every other candidate 

    vote_graph = nx.DiGraph()
    vote_graph.add_nodes_from(prof.candidates)
    vote_graph.add_weighted_edges_from([(c1,c2,prof.support(c1,c2)) if prof.support(c1,c2) > prof.support(c2,c1) else   (c1,c2,0)
           for c1 in prof.candidates 
           for c2 in prof.candidates if c1 != c2 ])
    
    beat_paths_weights = {c:{c2:0 for c2 in prof.candidates if c2 != c} for c in prof.candidates}
    for c in prof.candidates: 
        for other_c in beat_paths_weights[c].keys():
            beat_paths_weights[c][other_c] = max([min([vote_graph[p[i]][p[i+1]]['weight'] for i in range(0,len(p)-1)]) for p in nx.all_simple_paths(vote_graph, c, other_c)])
    
    winners = list()
    for c in prof.candidates: 
        if all([beat_paths_weights[c][c2] >= beat_paths_weights[c2][c] for c2 in prof.candidates  if c2 != c]):
            winners.append(c)

    return sorted(list(winners))


def ranked_pairs(prof):
    '''Ranked Pairs'''
    mg = generate_margin_graph(prof)
    wmg = generate_weak_margin_graph(prof)
    if len(find_condorcet_winner(mg)) != 0:
        winning_set = find_condorcet_winner(mg)
    else:
        winning_set = list()            
        margins = sorted(list(set([e[2]['weight'] for e in wmg.edges(data=True)])), reverse=True)
        sorted_edges = [[e for e in wmg.edges(data=True) if e[2]['weight'] == w] for w in margins]
        tbs = product(*[permutations(edges) for edges in sorted_edges])
        for tb in tbs:
            edges = flatten(tb)
            new_ranking = nx.DiGraph() 
            for e in edges: 
                new_ranking.add_edge(e[0], e[1], weight=e[2]['weight'])
                if  has_cycle(new_ranking):
                    new_ranking.remove_edge(e[0], e[1])
            winning_set.append(unbeaten_candidates(new_ranking)[0])
    return sorted(list(set(winning_set)))


# Implementation of Beat Path using the Floyd Warshall-Algorithm
def beatpath_faster(prof):   
    """Beat Path (faster)"""
    mg = [[-np.inf for _ in prof.candidates] for _ in prof.candidates]
    for c1 in prof.candidates:
        for c2 in prof.candidates:
            if (prof.support(c1,c2) > prof.support(c2,c1) or c1 == c2):
                mg[c1][c2] = prof.support(c1,c2) - prof.support(c2,c1)
    strength = map(lambda i : map(lambda j : j , i) , mg) 
    for i in prof.candidates:         
        for j in prof.candidates: 
            if i!= j:
                for k in prof.candidates: 
                    if i!= k and j != k:
                        strength[j][k] = max(strength[j][k], min(strength[j][i],strength[i][k]))
    winners = {i:True for i in prof.candidates}
    for i in prof.candidates: 
        for j in prof.candidates:
            if i!=j:
                if strength[j][i] > strength[i][j]:
                    winners[i] = False
    return sorted([c for c in prof.candidates if winners[c]])


# Implementation of Split Cycle by adapting the Floyd-Warshall Algorithm
def splitcycle_faster(prof):   
    """Split Cycle (faster)"""
    weak_condorcet_winners = {c:True for c in prof.candidates}
    mg = [[-np.inf for _ in prof.candidates] for _ in prof.candidates]
    for c1 in prof.candidates:
        for c2 in prof.candidates:
            if (prof.support(c1,c2) > prof.support(c2,c1) or c1 == c2):
                mg[c1][c2] = prof.support(c1,c2) - prof.support(c2,c1)
                weak_condorcet_winners[c2] = weak_condorcet_winners[c2] and (c1 == c2)
    
    strength = map(lambda i : map(lambda j : j , i) , mg) 
    for i in prof.candidates:         
        for j in prof.candidates: 
            if i!= j:
                if not weak_condorcet_winners[j]:
                    for k in prof.candidates: 
                        if i!= k and j != k:
                            strength[j][k] = max(strength[j][k], min(strength[j][i],strength[i][k]))
    winners = {i:True for i in prof.candidates}
    for i in prof.candidates: 
        for j in prof.candidates:
            if i!=j:
                if mg[j][i] > strength[i][j]:
                    winners[i] = False
    return sorted([c for c in prof.candidates if winners[c]])
