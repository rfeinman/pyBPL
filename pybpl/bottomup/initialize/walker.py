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
        self.clear()

    def clear(self):
        """
        Reset the walker.
        Clear the WalkerStroke list and set all nodes & edges as unvisited
        """
        self.list_ws = []
        nx.set_node_attributes(self.graph, False, name='visited')
        nx.set_edge_attributes(self.graph, False, name='visited')

    @property
    def ns(self):
        # number of strokes
        return len(self.list_ws)

    @property
    def S(self):
        # full trajectories for each stroke
        return [elt.stk for elt in self.list_ws]

    @property
    def visited_edges(self):
        eid_list = set()
        for eid in self.graph.edges():
            if self.graph.edges[eid]['visited']:
                eid_list.add(eid)
        return eid_list

    @property
    def unvisited_edges(self):
        eid_list = set()
        for eid in self.graph.edges():
            if not self.graph.edges[eid]['visited']:
                eid_list.add(eid)
        return eid_list

    @property
    def complete(self):
        # is the entire graph drawn?
        is_visited = [edge['visited'] for edge in self.graph.edges.values()]
        return all(is_visited)

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
        is_visited = np.array([self.graph.edges[eid]['visited'] for eid in vei])
        return cell_traj[~is_visited], vei[~is_visited]

    def select_new_moves(self, sel_new):
        _, vei = self.list_ws[-1].get_moves()
        is_visited = np.array([self.graph.edges[eid]['visited'] for eid in vei])
        findx = np.where(~is_visited)[0]
        sel = findx[sel_new]
        self.list_ws[-1].select_mov(sel)