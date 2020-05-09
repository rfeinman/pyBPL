import abc
import numpy as np
import networkx as nx

from .walker_stroke import WalkerStroke



class Walker(abc.ABC):
    """
    Walker: Class that controls a walk on the directed graph
    that is defined by the skeleton. The walk is complete
    when all of the edges have been covered.
    """
    def __init__(self, graph, image):
        """
        Parameters
        ----------
        graph : nx.Graph
            the (undirected) graph to walk on
        image : np.ndarray
            (H,W) original image in binary format
        """
        self.graph = graph
        self.image = image
        self.list_ws = []

    @property
    def ns(self):
        # number of strokes
        return len(self.list_ws)

    @property
    def S(self):
        # full trajectories for each stroke
        raise NotImplementedError

    @property
    def edges_visited(self):
        # is each edge visited?
        raise NotImplementedError

    @property
    def nodes_visited(self):
        # is each node visited?
        raise NotImplementedError

    @property
    def complete(self):
        # is the entire graph drawn?
        return np.all(self.edges_visited)

    @property
    def curr_ni(self):
        # current nodes index
        raise NotImplementedError

    @property
    def curr_pt(self):
        # current point location
        raise NotImplementedError

    def add_singletons(self):
        for nid in nx.isolates(self.graph):
            stroke = WalkerStroke(self.graph)
            stroke.start_pt = self.graph.nodes[nid]['o']
            self.list_ws.append(stroke)

    def get_moves(self):
        raise NotImplementedError

    def select_moves(self, sel):
        raise NotImplementedError

    def get_new_moves(self):
        raise NotImplementedError

    def select_new_moves(self, sel_new):
        raise NotImplementedError