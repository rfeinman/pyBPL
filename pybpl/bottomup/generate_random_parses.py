import warnings

from .skeleton import extract_skeleton
from .initialize import RandomWalker


def generate_random_parses(I, nwalk_det=5, max_nstroke=100, max_nwalk=150):
    warnings.warn("using incomplete python implementation of "
                  "'generate_random_walks' function")

    # get character skeleton from the fast bottom-up method
    graph = extract_skeleton(I)

    # initialize random walker
    walker = RandomWalker(graph)
    parses = []

    # add deterministic minimum-angle walks
    for i in range(nwalk_det):
        parses.append(walker.det_walk())

    # sample random walks until we reach capacity
    while num_strokes(parses) < max_nstroke and len(parses) < max_nwalk:
        list_walks = walker.sample(1)
        parses.append(list_walks[0])


def num_strokes(parses):
    raise NotImplementedError
