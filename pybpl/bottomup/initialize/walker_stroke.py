import numpy as np



class WalkerStroke:
    def __init__(self, graph, start_pt=None):
        # undirected graph
        self.graph = graph
        # (2,) stroke starting point. only necessary when ei is empty
        self.start_pt = start_pt
        # k-length list of edges, each of which is a tuple (u,v)
        self.ei = []
        # k-length list of booleans indicating whether to flip each edge
        self.ei_flip = []

    @property
    def curr_pt(self):
        if len(self.ei) == 0:
            return self.start_pt
        eid = self.ei[-1]
        assert eid in self.graph.edges
        edge = self.graph.edges[eid]
        traj = edge['pts']
        if self.ei_flip[-1]:
            traj = np.flip(traj, 0)
        return traj[-1]

    @property
    def curr_ni(self):
        """
        get index of current node, i.e. node where most recent edge ended
        """
        eid, flip = self.ei[-1], self.ei_flip[-1]
        if flip:
            return eid[0]
        else:
            return eid[1]
