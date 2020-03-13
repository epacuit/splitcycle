
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

def find_condorcet_losers(G):
    # edge case: there is no Condorcet loser if there is only one node
    if len(G.nodes) == 1:
        return []
    return [n for n in G.nodes if G.in_degree(n) == len(G.nodes) - 1]

def find_condorcet_loser(prof):
    
    G = generate_majority_graph(prof)
    # edge case: there is no Condorcet loser if there is only one node
    if len(G.nodes) == 1:
        return []
    return [n for n in G.nodes if G.in_degree(n) == len(G.nodes) - 1]

flatten = lambda l: [item for sublist in l for item in sublist]

## Voting Methods
def plurality(profile):
    """Plurality"""

    plurality_scores = profile.plurality_scores()
    max_plurality_score = max(plurality_scores.values())
    
    return sorted([c for c in plurality_scores.keys() if plurality_scores[c] == max_plurality_score])

def majority(profile):
    '''Majority'''
    
    maj_size = profile.strict_maj_size()
    
    plurality_scores = profile.plurality_scores()
    maj_winner = [c for c in profile.candidates if c in plurality_scores.keys() and plurality_scores[c] >= maj_size]
    
    return sorted(maj_winner)

def borda(profile):
    """Borda"""
    
    candidates = profile.candidates
    borda_scores = profile.borda_scores()
    max_borda_score = max(borda_scores.values())
    
    return sorted([c for c in candidates if borda_scores[c] == max_borda_score])


def plurality_with_runoff(profile):
    """PluralityWRunoff"""
    
    main_cands = {profile.cname(c): c for c in profile.candidates}
    plurality_scores = profile.plurality_scores()
    max_plurality_score = max(plurality_scores.values())
    first = [c for c in profile.candidates if plurality_scores[c] == max_plurality_score]
    if len(first) > 1:
        runoff_candidates = first
    else:
        second_plurality_score = list(reversed(sorted(plurality_scores.values())))[1]
        second = [c for c in profile.candidates if plurality_scores[c] == second_plurality_score]
        runoff_candidates = first + second
        
    runoff_profile = profile.remove_candidates([c for c in profile.candidates if c not in runoff_candidates])
    return [main_cands[runoff_profile.cname(c)] for c in plurality(runoff_profile)]

def hare(profile):
    """Ranked Choice"""
    
    candidates = profile.candidates
        
    plurality_scores = profile.plurality_scores()
    
    winners = majority(profile)
    main_cands = {profile.cname(c): c for c in profile.candidates}
    
    updated_profile = profile
    while len(winners) == 0:
        
        min_plurality_score = min(plurality_scores.values())
        lowest_first_place_votes = [c for c in plurality_scores.keys() if plurality_scores[c] == min_plurality_score]
        
        lowest_first_place_votes_orig_names = [main_cands[updated_profile.cname(c)] for c in lowest_first_place_votes]
        # remove lowest plurality winners
        updated_profile = updated_profile.remove_candidates(lowest_first_place_votes)        
        if updated_profile.is_empty():
            winners = sorted(lowest_first_place_votes_orig_names)
        else:
            plurality_scores = updated_profile.plurality_scores()
            winners = sorted([main_cands[updated_profile.cname(c)] for c in majority(updated_profile)])
     
    return winners

def coombs(profile):
    """Coombs"""
    
    winners = majority(profile)
    
    updated_profile = profile
    
    main_cands = {profile.cname(c): c for c in profile.candidates}

    while len(winners) == 0:
        
        candidates, num_candidates = updated_profile.candidates, updated_profile.num_candidates
        last_place_scores = {c: updated_profile.num_rank(c,level=num_candidates) for c in candidates}
        max_last_place_score = max(last_place_scores.values())
        greatest_last_place_votes = [c for c in candidates if c in last_place_scores.keys() and last_place_scores[c] == max_last_place_score]
        # remove lowest plurality winners

        greatest_last_place_votes_orig_names = [main_cands[updated_profile.cname(c)] for c in greatest_last_place_votes]

        updated_profile = updated_profile.remove_candidates(greatest_last_place_votes)

        if updated_profile.is_empty():
            winners = sorted(greatest_last_place_votes_orig_names)
        else:
            winners = [main_cands[updated_profile.cname(c)] for c in majority(updated_profile)]

    return winners


def strict_nanson(profile):
    """StrictNanson"""
    
    borda_scores = profile.borda_scores()
    avg_borda_scores = np.mean(borda_scores.values())
    below_borda_avg_candidates = [c for c in profile.candidates if borda_scores[c] < avg_borda_scores]
    
    updated_profile = profile.remove_candidates(below_borda_avg_candidates)
    main_cands = {profile.cname(c): c for c in profile.candidates}

    winners = updated_profile.candidates if updated_profile.num_candidates == 1 else []
    
    while len(winners) == 0: 
        
        borda_scores = updated_profile.borda_scores()
        avg_borda_scores = np.mean(borda_scores.values())
    
        below_borda_avg_candidates = [c for c in updated_profile.candidates if borda_scores[c] < avg_borda_scores]
        updated_profile = updated_profile.remove_candidates(below_borda_avg_candidates)
        
        if len(below_borda_avg_candidates) == 0:
            winners = sorted(updated_profile.candidates)
        elif updated_profile.num_candidates == 1:
            winners = updated_profile.candidates
        
    return [main_cands[updated_profile.cname(c)] for c in winners]

def weak_nanson(profile):
    """WeakNanson"""
    
    borda_scores = profile.borda_scores()
    avg_borda_scores = np.mean(borda_scores.values())
    below_borda_avg_candidates = [c for c in profile.candidates if borda_scores[c] <= avg_borda_scores]
    main_cands = {profile.cname(c): c for c in profile.candidates}
    
    updated_profile = profile.remove_candidates(below_borda_avg_candidates)

    winners = updated_profile.candidates if updated_profile.num_candidates == 1 else []
    
    if len(below_borda_avg_candidates) == profile.num_candidates: 
        winners = sorted(profile.candidates)
    else:
        while len(winners) == 0: 
        
            borda_scores = updated_profile.borda_scores()
            avg_borda_scores = np.mean(borda_scores.values())

            below_borda_avg_candidates = [c for c in updated_profile.candidates if borda_scores[c] <= avg_borda_scores]
            below_borda_avg_orig_names = [main_cands[updated_profile.cname(c)] for c in below_borda_avg_candidates]
       
            updated_profile = updated_profile.remove_candidates(below_borda_avg_candidates)

            if updated_profile.num_candidates  == 0:
                winners = sorted(below_borda_avg_orig_names)
            elif updated_profile.num_candidates == 1:
                winners = [main_cands[updated_profile.cname(c)] for c in updated_profile.candidates] 
        
    return winners 

def baldwin(profile):
    """Baldwin"""

    borda_scores = profile.borda_scores()
    candidates = profile.candidates
    main_cands = {profile.cname(c): c for c in profile.candidates}
    min_borda_score = min(borda_scores.values())
    last_place_borda_scores = [c for c in candidates if c in borda_scores.keys() and borda_scores[c] == min_borda_score]
        
    # remove lowest plurality winners
    updated_profile = profile.remove_candidates(last_place_borda_scores)
    
    winners = list()
    if updated_profile.num_candidates == 0: 
        winners = sorted(last_place_borda_scores)
    
    while len(winners) == 0:
        
        borda_scores = updated_profile.borda_scores()
                
        candidates = updated_profile.candidates
        min_borda_score = min(borda_scores.values())
        last_place_borda_scores = [c for c in candidates if c in borda_scores.keys() and borda_scores[c] == min_borda_score]
        last_place_borda_orig_names = [main_cands[updated_profile.cname(c)] for c in last_place_borda_scores]
        
        # remove lowest borda scores
        updated_profile = updated_profile.remove_candidates(last_place_borda_scores)
        
        if updated_profile.is_empty():
            winners = sorted(last_place_borda_orig_names)
        elif updated_profile.num_candidates == 1:
            winners = [main_cands[updated_profile.cname(c)] for c in updated_profile.candidates]
    
    return winners 

def condorcet(profile):
    """Condorcet"""
    
    tally = profile.tally()
    cond_winner = filter(lambda c: all([tally[c][_] > 0 for _ in tally[c].keys()]), profile.candidates)
    return sorted(cond_winner) if len(cond_winner) > 0 else sorted(profile.candidates)

def copeland(profile): 
    """Copeland"""
    
    tally  = profile.tally()
    candidates = tally.keys()
    copeland_scores = {c:len([1 for c2 in tally[c].keys() if tally[c][c2] > 0]) - len([1 for c2 in tally[c].keys() if tally[c][c2] < 0]) for c in candidates}
    max_copeland_score = max(copeland_scores.values())
    return sorted([c for c in candidates if copeland_scores[c] == max_copeland_score])

def minimax(profile, score_method="winning"):
    """MiniMax"""
    
    candidates = profile.candidates
    
    score_functions = {
        "winning": lambda c1,c2: profile.support(c1,c2) if profile.support(c1,c2) > profile.support(c2,c1) else 0,
        "margins": lambda c1,c2: profile.support(c1,c2)   -  profile.support(c2,c1),
        "pairwise_opposition": lambda c1,c2: profile.support(c1,c2)
    } 
    scores = {c: max([score_functions[score_method](_c,c) for _c in candidates if _c != c]) for c in candidates}
    min_score = min(scores.values())
    return sorted([c for c in candidates if scores[c] == min_score])


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

###
#Helper functions
###

def create_tie_breaker(lin_order):
    # given a linear ordering, create a tie breaker function
    
    return lambda cands: sorted(list(cands), key=lambda c: lin_order.index(c))[0]

def create_all_tie_breakers(candidate_names):
    
    # create all tie breakers for a set of candidates 
    # (each linear ordering gives a different tie-breaker)
    
    linear_orders = generate_linear_orderings(candidate_names)
    
    return [create_tie_breaker(lo) for lo in linear_orders]
    
def tie_breaker_cand_names(winners):
    
    # use the tie breaking that returns the candidate
    # whose name is alphabetically  earlies
    
    return sorted(winners)[0]

def tie_breaker_random(winners):
    
    # break time by  uniform probability over the winners
    return random.choice(winners)


def vms_to_str(voting_methods):
    # return a string of the voting method names
    return ','.join([_.__doc__ for _ in voting_methods])

def dom_to_str(dom):
    # return a string of the dominance tuple (weak_dom_name, strict_dom_name)
    return '|'.join(list(dom))

def str_to_dom(dom_str):
    
    return tuple(dom_str.split("|"))
    
def same_voting_methods_string(vms_str1, vms_str2):
    
    # test if the vothing methods strings are the same, e.g., 
    # the following should return true: "Borda", " Borda"; 
    # "Borda, Plurality", "Plurality, Borda"; and 
    # "Borda, Plurality, Plurality", "Plurality, Borda"
    vms1 = [vm_str.strip() for vm_str in vms_str1.split(",")]
    vms2 = [vm_str.strip() for vm_str in vms_str2.split(",")]
    
    return set(vms1) == set(vms2)
