'''Speed test for comparing DFS and BFS performance'''

import time
import splitcycle

# generate random ballots
N_BALLOTS = 100
N_CANDIDATES = 20
N_DIM = 2
N_TRIALS = 10000
ONLY_MULTIPLE = False

ballot_list = []

# alphabet soup of candidate names
candidates = [chr(i) for i in range(65, 65 + N_CANDIDATES)]

for i in range(N_TRIALS):
    ballot_list.append(splitcycle.utils.gen_random_ballots(
        N_BALLOTS, N_CANDIDATES, model=f'euclidean-{N_DIM}'
    ))

SUM_TIMES_DFS = 0
COUNT = 0

# dfs
for i in range(N_TRIALS):
    start = time.time()
    winners = splitcycle.elect(ballot_list[i], candidates, True)
    end = time.time()
    if ONLY_MULTIPLE and len(winners) == 1:
        continue
    SUM_TIMES_DFS += end - start
    COUNT += 1

SUM_TIMES_BFS = 0

# bfs
for i in range(N_TRIALS):
    start = time.time()
    winners = splitcycle.elect(ballot_list[i], candidates, False)
    end = time.time()
    if ONLY_MULTIPLE and len(winners) == 1:
        continue
    SUM_TIMES_BFS += end - start
    COUNT += 1

print(f'DFS avg time: {1000 * SUM_TIMES_DFS / COUNT} ms')
print(f'BFS avg time: {1000 * SUM_TIMES_BFS / COUNT} ms')

if SUM_TIMES_BFS < SUM_TIMES_DFS:
    print('BFS is faster')
else:
    print('DFS is faster')
