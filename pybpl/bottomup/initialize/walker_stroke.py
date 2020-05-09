class WalkerStroke:
    def __init__(self, graph):
        self.graph = graph
        self.start_pt = None
        self.ei = []
        self.ei_flip = []

    @property
    def k(self):
        return len(self.ei)