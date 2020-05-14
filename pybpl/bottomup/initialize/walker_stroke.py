import numpy as np



class WalkerStroke:
    def __init__(self, graph, start_ni):
        # undirected graph
        self.graph = graph
        # list of visited nodes
        self.list_ni = [start_ni]

    @property
    def curr_ni(self):
        return self.list_ni[-1]

    def move(self, next_ni):
        self.list_ni.append(next_ni)
