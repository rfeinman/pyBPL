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
        pt = self.curr_pt
        return map_pt_to_ni(self.graph, pt)

def map_pt_to_ni(graph, pt):
    mindist = float('inf')
    nid = -1
    for i, node in graph.nodes.items():
        dist = np.linalg.norm(node['o'] - pt)
        if dist < mindist:
            nid = i
            mindist = dist
    assert nid != -1
    return nid
