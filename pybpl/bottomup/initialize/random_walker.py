import numpy as np
from scipy.special import logsumexp
import networkx as nx

from ..parameters import ParametersBottomup
from .walker import Walker
from .walker_stroke import WalkerStroke
from .fit_smooth_stk import fit_smooth_stk



class RandomWalker(Walker):
    """
    RandomWalker : produce a random walk on the graph skeleton.
    It is presumed the direction and order of the strokes doesn't
    matter, and this will be optimized later.
    """
    def __init__(self, graph, image, ps=None):
        """
        Parameters
        ----------
        graph : nx.Graph
            the (undirected) graph to walk on
        image : np.ndarray
            (H,W) original image in binary format
        ps : ParametersBottomup
            parameters for bottom-up methods
        """
        super().__init__(graph, image)
        if ps is None:
            ps = ParametersBottomup()
        self.ps = ps
        self.verbose = None
        self.exp_wt_start = None
        self.lambda_softmax = None

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
        Place your pen down at an unvisited node,
        inversely proportional to the number of unvisited
        edges going from it.
        """
        new_pts, degree = self._pts_on_new_edges
        logwts = self.exp_wt_start * np.log(1/degree)
        logwts = logwts - logsumexp(logwts)
        wts = np.exp(logwts)
        rindx = np.random.choice(len(wts), p=wts)
        stroke = WalkerStroke(self.graph, start_pt=new_pts[rindx])
        self.list_ws.append(stroke)
        if not self.complete:
            self.pen_simple_step()

    def pen_angle_step(self):
        """
        Angle move: select a step based on the angle
        from the current trajectory.
        """
        cell_traj, vei = self.get_moves()
        n = len(vei)
        if n == 0:
            self.pen_up_down()
            return

        # get angles for all edges
        is_visited = np.array([self.graph.edges[eid]['visited'] for eid in vei])
        angles = self.ps.faux_angle_repeat * np.ones(n) # default angle for used edges
        angles[~is_visited] = self._angles_for_moves(cell_traj[~is_visited])
        angles = np.append(angles, self.ps.faux_angle_lift)

        # select move stochastically
        rindx = self._action_via_angle(angles)
        if rindx == (len(angles)-1):
            self.pen_up_down()
        else:
            self.select_moves(rindx)

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
            return
        sel = np.random.randint(n)
        self.select_new_moves(sel)

    def _action_via_angle(self, angles):
        """
        Given a vector of angles, compute move probabilities proportional
        to exp(-lambda*angle/180) and sample a move index
        """
        theta = angles / 180
        netinput = -self.lambda_softmax*theta
        logpvec = netinput - logsumexp(netinput)
        pvec = np.exp(logpvec)
        rindx = np.random.choice(len(pvec), p=pvec)
        return rindx

    def _angles_for_moves(self, cell_traj):
        """
        Compute angle for each move.
        """
        junct_pt = self.curr_pt
        nt = len(cell_traj)

        # for each possible move, list the entire stroke that we
        # would create if we accepted it
        last_stk = self.S[-1]
        cell_prop = [np.concatenate([last_stk, traj[1:]]) for traj in cell_traj]

        # smooth each candidate stroke
        cell_smooth = [fit_smooth_stk(prop, self.image, self.ps) for prop in cell_prop]

        # at the junction, isolate the relevant segments of the
        # smoothed stroke
        angles = np.zeros(nt)
        for i in range(nt):
            first_half, second_half = \
                split_by_junction(junct_pt, cell_smooth[i], self.ps.rad_junction)
            angles[i] = compute_angle(second_half, first_half, self.ps)

        if np.any(np.isnan(angles) | (np.imag(angles)!= 0)):
            raise Exception('error in angle calculation')

        return angles

    @property
    def _pts_on_new_edges(self):
        """
        For all new edges in the graph, make a list of their start/end
        points where we may want to drop our pen. Also, return their degree
        """
        new_eids = self.unvisited_edges
        new_nids = []
        for eid in new_eids:
            new_nids.extend([eid[0], eid[1]])

        # degree for each node is the total number of edges minus the
        # number of 'visited' edges
        degree = np.zeros(len(new_nids))
        for i, nid in enumerate(new_nids):
            degree[i] = self.graph.degree(nid) - self.graph.degree(nid, weight='visited')
        list_pts = [self.graph.nodes(nid)['o'] for nid in new_nids]

        return list_pts, degree

def split_by_junction(junct_pt, traj, radius):
    """
    Get portion of trajectory within the specific radius,
    and divide it in two based on the closest point to the junction
    """
    raise NotImplementedError

def compute_angle(seg_ext, seg_prev, ps):
    """
    Compute the angle between two vectors
    """
    raise NotImplementedError
