'''
    File: generate_margin_graphs.py
    Author: Eric Pacuit (epacuit@umd.edu)
    Date: September 12, 2021
    
    Functions to generate a margin graph
    
'''


import networkx as nx
import random

def generate_random_edge_ordered_tournament(num_cands, is_uniquely_weighted = True): 
    
    
    mg = nx.DiGraph()
    mg.add_nodes_from(list(range(num_cands)))
    
    for c1 in mg.nodes: 
        for c2 in mg.nodes: 
            if c1 != c2: 
                if not mg.has_edge(c1, c2) and not mg.has_edge(c2,c1):
                    if random.choice([0,1])  == 0: 
                        mg.add_edge(c1,c2, weight=-1)
                    else: 
                        mg.add_edge(c2,c1, weight=-1)
                   
    edges = [e for e in mg.edges(data=True)]
    edge_indices = list(range(len(mg.edges)))
    random.shuffle(edge_indices)
    
    for i, e_idx in enumerate(edge_indices): 
        edges[e_idx][2]['weight']  = 2 * (i+1)
    
   
    return mg

