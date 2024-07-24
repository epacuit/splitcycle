from splitcycle import utils

pytest_plugins = ("benchmark",)


def test_euclid(benchmark):
    benchmark(utils.gen_random_ballots, 1000, 10, model="euclidean-6")


def test_ic(benchmark):
    benchmark(utils.gen_random_ballots, 1000, 10, model="ic")


def test_ic_ties(benchmark):
    benchmark(utils.gen_random_ballots, 1000, 10)
