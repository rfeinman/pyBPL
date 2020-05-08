from .walker import Walker



class RandomWalker(Walker):
    def __init__(self, T, ps=None):
        super().__init__(T)
        self.ps = ps
        self.verbose = None
        self.exp_wt_start = None
        self.lambda_softmax = None

    def sample(self, nsamp):
        """
        Produce "nsamp" samples from the random walk model
        """
        pass

    def det_walk(self):
        """
        Produce a deterministic walk
        """
        pass

    def make(self, verbose):
        """
        Make a random walk through the graph
        """
        pass

    def clear(self):
        """
        Clear the object.
        """
        self.list_WS = []

    def pen_up_down(self):
        """
        Place your pen down at an unvisited edge,
        inversely proportional to the number of unvisited
        paths going from it.
        """
        pass

    def pen_angle_step(self):
        """
        Angle move: select a step based on the angle
        from the current trajectory.
        """
        pass

    def pen_simple_step(self):
        """
        Simple move: select a step uniformly at random
        from the step of new edges. Do not consider lifting
        the pen until you run out of new edges.
        """
        pass