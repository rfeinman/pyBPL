import numpy as np
from scipy.special import logsumexp
import networkx as nx

from .walker import Walker
from .walker_stroke import WalkerStroke
from ..parameters_bottomup import defaultps_bottomup



class RandomWalker(Walker):
    """
    RandomWalker : produce a random walk on the graph skeleton.
    It is presumed the direction and order of the strokes doesn't
    matter, and this will be optimized later.
    """
    def __init__(self, graph, ps=None):
        """
        Parameters
        ----------
        graph : nx.Graph
            the (undirected) graph to walk on
        ps : defaultps_bottomup
            parameters for bottom-up methods
        """
        super().__init__(graph)
        if ps is None:
            ps = defaultps_bottomup()
        self.ps = ps
        self.verbose = None
        self.exp_wt_start = None
        self.lambda_softmax = None

    def clear(self):
        """
        Clear the object.
        """
        self.list_ws = []

    def sample(self):
        """
        Produce a sample from the random walk model
        """
        self.exp_wt_start = np.random.choice(self.ps.int_exp_wt)
        self.lambda_softmax = np.random.choice(self.ps.int_lambda_soft)
        myns = float('inf')
        while myns > self.ps.max_len:
            walk = self.make()
            myns = len(walk)
        return walk

    def det_walk(self):
        """
        Produce a deterministic walk
        """
        self.exp_wt_start = 1000
        self.lambda_softmax = 1000
        walk = self.make()
        return walk

    def make(self):
        """
        Make a random walk through the graph
        """
        self.clear()
        self.add_singletons()
        if not self.complete:
            self.pen_up_down()
        while not self.complete:
            self.pen_angle_step()
        walk = self.S
        return walk

    def pen_up_down(self):
        """
        Place your pen down at an unvisited edge,
        inversely proportional to the number of unvisited
        paths going from it.
        """
        new_pts, degree = self._pts_on_new_edges
        logwts = self.exp_wt_start * np.log(1/degree)
        logwts = logwts - logsumexp(logwts)
        wts = np.exp(logwts)
        rindx = np.random.choice(len(wts), p=wts)
        stroke = WalkerStroke(self.graph)
        stroke.start_pt = new_pts[rindx]
        self.list_ws.append(stroke)
        if not self.complete:
            self.pen_simple_step()

    def pen_angle_step(self):
        """
        Angle move: select a step based on the angle
        from the current trajectory.
        """
        raise NotImplementedError

    def viz_angle_step(self, angles_for_new, first_half, second_half, cell_smooth):
        raise NotImplementedError

    def pen_simple_step(self):
        """
        Simple move: select a step uniformly at random
        from the step of new edges. Do not consider lifting
        the pen until you run out of new edges.
        """
        _, vei = self.get_new_moves()
        n = len(vei)
        if n == 0:
            self.pen_up_down()
        else:
            sel = np.random.randint(n)
            self.select_new_moves(sel)

    def _action_via_angle(self, angles):
        raise NotImplementedError

    def _angles_for_moves(self, cell_traj):
        raise NotImplementedError

    @property
    def _pts_on_new_edges(self):
        raise NotImplementedError