
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


# Splity Cycle  
def split_cycle(prof):
    """Split Cycle"""
    
    # create the margin graph
    mg = generate_margin_graph(prof) 

    # find the cycle numbers (Definition  3.12) for each candidate
    cycle_number = {cs:0 for cs in permutations(prof.candidates,2)}
    for cycle in nx.simple_cycles(mg): # for each cycle in the margin graph

        # get all the margins (i.e., the weights) of the edges in the cycle
        margins = list() 
        for idx,c1 in enumerate(cycle): 
            next_idx = idx + 1 if (idx + 1) < len(cycle) else 0
            c2 = cycle[next_idx]
            margins.append(mg[c1][c2]['weight'])
            
        split_number = min(margins) # the split number of the cycle (Definition 3.2)
        for c1,c2 in cycle_number.keys():
            c1_index = cycle.index(c1) if c1 in cycle else -1
            c2_index = cycle.index(c2) if c2 in cycle else -1

            # only need to check cycles with an edge from c1 to c2
            if (c1_index != -1 and c2_index != -1) and ((c2_index == c1_index + 1) or (c1_index == len(cycle)-1 and c2_index == 0)):
                cycle_number[(c1,c2)] = split_number if split_number > cycle_number[(c1,c2)] else cycle_number[(c1,c2)]        

    # construct the defeat relation, where a defeats b if margin(a,b) > cycle_number(a,b) (see Lemma 3.13)
    defeat = nx.DiGraph()
    defeat.add_nodes_from(prof.candidates)
    defeat.add_edges_from([(c1,c2)  
           for c1 in prof.candidates 
           for c2 in prof.candidates if c1 != c2 if prof.margin(c1,c2) > cycle_number[(c1,c2)]])

    # the winners are candidates not defeated by any other candidate
    winners = unbeaten_candidates(defeat)
    
    return sorted(list(set(winners)))

# Return Split Cycle winners, split numbers, and defeat graph
def split_cycle_with_data(prof):
    """Split Cycle (with data)"""
    
    # create the margin graph
    mg = generate_margin_graph(prof) 

    # find the cycle numbers (Definition  3.12) for each candidate
    cycle_number = {cs:0 for cs in permutations(prof.candidates,2)}
    split_numbers = dict()
    for cycle in nx.simple_cycles(mg): # for each cycle in the margin graph

        # get all the margins (i.e., the weights) of the edges in the cycle
        margins = list() 
        for idx,c1 in enumerate(cycle): 
            next_idx = idx + 1 if (idx + 1) < len(cycle) else 0
            c2 = cycle[next_idx]
            margins.append(mg[c1][c2]['weight'])
            
        split_number = min(margins) # the split number of the cycle (Definition 3.2)
        
        split_numbers.update({tuple(cycle): split_number}) # record split numbers
        for c1,c2 in cycle_number.keys():
            c1_index = cycle.index(c1) if c1 in cycle else -1
            c2_index = cycle.index(c2) if c2 in cycle else -1

            # only need to check cycles with an edge from c1 to c2
            if (c1_index != -1 and c2_index != -1) and ((c2_index == c1_index + 1) or (c1_index == len(cycle)-1 and c2_index == 0)):
                cycle_number[(c1,c2)] = split_number if split_number > cycle_number[(c1,c2)] else cycle_number[(c1,c2)]        

    # construct the defeat relation, where a defeats b if margin(a,b) > cycle_number(a,b) (see Lemma 3.13)
    defeat = nx.DiGraph()
    defeat.add_nodes_from(prof.candidates)
    defeat.add_edges_from([(c1,c2)  
           for c1 in prof.candidates 
           for c2 in prof.candidates if c1 != c2 if prof.margin(c1,c2) > cycle_number[(c1,c2)]])

    # the winners are candidates not defeated by any other candidate
    winners = unbeaten_candidates(defeat)
    
    return sorted(list(set(winners))), split_numbers, defeat

# Display all the Split Cycle data
def display_split_cycle_data(prof):
    
    sc_winners, split_numbers, defeat = split_cycle_with_data(prof)

    print ""
    display_winners(split_cycle, prof)
    print "\n---\n"
    print "Split numbers\n"
    for cycle in split_numbers.keys():
        print "Split#({}) = {}".format(",".join([prof.cname(_) for _ in cycle]), split_numbers[cycle])
    
    print "\n---\n"
    print "Defeat relation"
    defeat = nx.relabel_nodes(defeat, {c:prof.cname(c) for c in defeat.nodes})
    pos = nx.circular_layout(defeat)
    nx.draw(defeat, pos, font_size=20, node_color='blue', font_color='white',node_size=700, width=1, lw=1.75, with_labels=True)
    
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
