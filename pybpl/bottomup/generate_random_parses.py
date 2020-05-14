import warnings

from .skeleton import extract_skeleton
from .initialize import RandomWalker



def generate_random_parses(image, nwalk_det=5, max_nstroke=100, max_nwalk=150):
    warnings.warn("using untested python implementation of "
                  "'generate_random_walks'")

    # get character skeleton from the fast bottom-up method
    graph = extract_skeleton(image)

    # initialize random walker and empty parse list
    walker = RandomWalker(graph, image)
    parses = []

    # add deterministic minimum-angle walks
    for i in range(nwalk_det):
        parses.append(walker.det_walk())

    # sample random walks until we reach capacity
    nwalk = len(parses)
    nstroke = sum([len(parse) for parse in parses])
    while nstroke < max_nstroke and nwalk < max_nwalk:
        walk = walker.sample()
        parses.append(walk)
        nwalk += 1
        nstroke += len(walk)

    return parses
