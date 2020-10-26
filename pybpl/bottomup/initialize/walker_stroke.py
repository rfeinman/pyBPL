import numpy as np



class WalkerStroke:
    def __init__(self, start_ni):
        self.list_ni = [start_ni]  # list of visited nodes
        self.list_ei = []  # list of visited edges

    @property
    def curr_ni(self):
        return self.list_ni[-1]

    def move(self, next_ei):
        assert isinstance(next_ei, tuple)
        next_ni = next_ei[1]
        self.list_ni.append(next_ni)
        self.list_ei.append(next_ei)
