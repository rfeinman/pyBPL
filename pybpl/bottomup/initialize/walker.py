from abc import ABCMeta
import numpy as np
import networkx as nx

from .walker_stroke import WalkerStroke



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
        self.list_ws = []

    @property
    def ns(self):
        # number of strokes
        return len(self.list_ws)

    @property
    def S(self):
        # full trajectories for each stroke
        return [elt.stk for elt in self.list_ws]

    @property
    def complete(self):
        # is the entire graph drawn?
        edges_visited = [edge['visited'] for edge in self.graph.edges.values()]
        return all(edges_visited)

    @property
    def curr_ni(self):
        # current nodes index
        return self.list_ws[-1].curr_ni

    @property
    def curr_pt(self):
        # current point location
        nid = self.curr_ni
        return self.graph.nodes[nid]['o']

    def add_singletons(self):
        for nid in nx.isolates(self.graph):
            start_pt = self.graph.nodes[nid]['o']
            stroke = WalkerStroke(self.graph, start_pt=start_pt)
            self.list_ws.append(stroke)

    def get_moves(self):
        return self.list_ws[-1].get_moves()

    def select_moves(self, sel):
        self.list_ws[-1].select_move(sel)

    def get_new_moves(self):
        cell_traj, vei = self.list_ws[-1].get_moves()
        visited = np.array([self.graph.edges[eid]['visited'] for eid in vei])
        return cell_traj[~visited], vei[~visited]

    def select_new_moves(self, sel_new):
        _, vei = self.list_ws[-1].get_moves()
        visited = np.array([self.graph.edges[eid]['visited'] for eid in vei])
        findx = np.where(~visited)[0]
        sel = findx[sel_new]
        self.list_ws[-1].select_mov(sel)