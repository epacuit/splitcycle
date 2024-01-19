'''Core utilities for SplitCycle package'''

from . import np

def is_square(A):
  '''Check if a given 2D matrix `A` is square'''
  return A.shape[0] == A.shape[1]

def has_strong_path(A, source, target, k):
  '''
  Given a square matrix `A`, return `True` if there is a path from `source`
  to `target` in the associated directed graph, where each edge has a weight
  greater than or equal to `k`, and `False` otherwise.

  '''
  if not is_square(A):
    raise TypeError('''`A` must be a square matrix.
`A` represents a directed graph as a square matrix, where `A[i][j]` represents the weight of the connection between node `i` and node `j`.
As all preferences are compared to each other, this matrix should include weights between any two nodes.
''')
  n = A.shape[0]
  visited = np.zeros(n, dtype=bool)

  def dfs(node):
    '''
    Depth-first search implementation:
    Traverse through `A` until a path between the node and target are found or until all nodes are searched.
    '''
    if node == target:
      return True
    
    visited[node] = True
    for neighbor, weight in enumerate(A[node, :]):
      if weight >= k and not visited[neighbor]:
        if dfs(neighbor):
          return True

    return False

  return dfs(source)

def split_cycle(margins, candidates=None, strength_function=None):
    '''
    If x has a positive margin over y and there is no path from y back to x of
    strength at least the margin of x over y, then x defeats y. The candidates
    that are undefeated are the Split Cycle winners.

    `margins`: a square matrix with margins of victory (positive) or defeat
               (negative) between candidates on its first axis and their
               opponents on the second
    
    Returns a sorted list of all SplitCycle winners
    '''
    if not is_square(margins):
      raise TypeError('''`margins` must be a square matrix.
`margins` represents a directed graph as a square matrix, where `margins[i][j]` represents the weight of the connection between node `i` and node `j`.
As all preferences are compared to each other, this matrix should include weights between any two nodes.
''')    
    candidates = range(margins.shape[0]) if candidates is None else candidates  
    strength_function = lambda a, b: margins[a, b] if strength_function is None \
      else strength_function

    potential_winners = set(candidates)

    for a in candidates:
      for b in candidates:
        if strength_function(a, b) < 0 and not has_strong_path(margins, a, b, strength_function(a, b)):
          potential_winners.discard(a)
          break

    return sorted(potential_winners)
