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
        ni = self.curr_ni
        return self.graph.nodes[ni]['o']

    def add_singletons(self):
        for ni in nx.isolates(self.graph):
            stroke = WalkerStroke(self.graph, start_ni=ni)
            self.list_ws.append(stroke)

    def get_moves(self):
        curr_ni = self.list_ws[-1].curr_ni
        list_ni = []
        for next_ni in self.graph.neighbors(curr_ni):
            list_ni.append(next_ni)
        return list_ni

    def get_new_moves(self):
        curr_ni = self.list_ws[-1].curr_ni
        list_ni = []
        for next_ni in self.graph.neighbors(curr_ni):
            if self.graph.edges[curr_ni,next_ni]['visited']:
                continue
            list_ni.append(next_ni)
        return list_ni

    def select_move(self, next_ni):
        curr_ni = self.list_ws[-1].curr_ni
        self.list_ws[-1].move(next_ni)
        self.graph.edges[curr_ni, next_ni]['visited'] = True
