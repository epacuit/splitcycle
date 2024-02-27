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

sum_times_dfs = 0
count = 0

# dfs
for i in range(N_TRIALS):
    start = time.time()
    winners = splitcycle.elect(ballot_list[i], candidates, True)
    end = time.time()
    if ONLY_MULTIPLE and len(winners) == 1:
        continue
    sum_times_dfs += end - start
    count += 1

sum_times_bfs = 0

# bfs
for i in range(N_TRIALS):
    start = time.time()
    winners = splitcycle.elect(ballot_list[i], candidates, False)
    end = time.time()
    if ONLY_MULTIPLE and len(winners) == 1:
        continue
    sum_times_bfs += end - start
    count += 1

print(f'DFS avg time: {1000 * sum_times_dfs / count} ms')
print(f'BFS avg time: {1000 * sum_times_bfs / count} ms')

if sum_times_bfs < sum_times_dfs:
    print('BFS is faster')
else:
    print('DFS is faster')
