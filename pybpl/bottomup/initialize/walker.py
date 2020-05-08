import numpy as np



class Walker:
    def __init__(self, T):
        self.T = T
        self.list_WS = []

    @property
    def ns(self):
        # number of strokes
        return len(self.list_WS)

    @property
    def S(self):
        # full trajectories for each stroke
        return

    @property
    def edges_visited(self):
        # is each edge visited?
        return

    @property
    def nodes_visited(self):
        # is each node visited?
        return

    @property
    def complete(self):
        # is the entire graph drawn?
        return np.all(self.edges_visited)

    @property
    def curr_ni(self):
        # current nodes index
        return

    @property
    def curr_pt(self):
        # current point location
        return