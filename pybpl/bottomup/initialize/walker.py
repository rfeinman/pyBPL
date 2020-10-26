from abc import ABCMeta
import numpy as np
import networkx as nx

from .walker_stroke import WalkerStroke
from . import util



class Walker(metaclass=ABCMeta):
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
        self.clear()

    def clear(self):
        """
        Reset the walker.
        Clear the WalkerStroke list and set all nodes & edges as unvisited
        """
        self.list_ws = []
        nx.set_edge_attributes(self.graph, False, name='visited')

    def visit(self, edge):
        self.graph.edges[edge]['visited'] = True

    def is_visited(self, edge):
        return self.graph.edges[edge]['visited']

    def is_unvisited(self, edge):
        return not self.is_visited(edge)

    def edges(self, *args, **kwargs):
        if isinstance(self.graph, nx.MultiGraph):
            return self.graph.edges(*args, **kwargs, keys=True)
        else:
            return self.graph.edges(*args, **kwargs)

    @property
    def ns(self):
        # number of strokes
        return len(self.list_ws)

    @property
    def S(self):
        # full trajectories for each stroke
        fn = lambda stk : util.stroke_from_nodes(self.graph, stk.list_ni)
        return list(map(fn, self.list_ws))

    @property
    def complete(self):
        # is the entire graph drawn?
        return all(map(self.is_visited, self.edges()))

    @property
    def curr_ni(self):
        # current nodes index
        return self.list_ws[-1].curr_ni

    @property
    def curr_pt(self):
        # current point location
        ni = self.curr_ni
        return self.graph.nodes[ni]['o']

    def add_singletons(self):
        for ni in nx.isolates(self.graph):
            stroke = WalkerStroke(ni)
            self.list_ws.append(stroke)

    def get_moves(self):
        curr_ni = self.list_ws[-1].curr_ni
        list_ni = []
        for edge in self.edges(curr_ni):
            list_ni.append(edge[1])
        return list_ni

    def get_new_moves(self):
        curr_ni = self.list_ws[-1].curr_ni
        list_ni = []
        for edge in filter(self.is_unvisited, self.edges(curr_ni)):
            list_ni.append(edge[1])
        return list_ni

    def select_move(self, next_ni):
        curr_ni = self.list_ws[-1].curr_ni
        self.list_ws[-1].move(next_ni)
        edge = (curr_ni, next_ni)
        self.visit(edge)
