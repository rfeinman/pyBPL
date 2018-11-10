class ConceptType(object):

    def __init__(self, k, P, R):
        self.k = k
        self.part_types = P
        self.relation_types = R

    def parameters(self):
        '''
        return list of parameters
        '''
        pass

    def train(self):
        '''
        makes params require grad
        '''
        pass

    def eval(self):
        '''
        makes params require no grad
        '''
        pass

    def to(self, device):
        '''
        moves parameters to device
        '''
        pass


class ConceptToken(object):

    def __init__(self, P, R):
        self.part_tokens = P
        self.relation_tokens = R

    def parameters(self):
        '''
        return list of parameters
        '''
        pass

    def train(self):
        '''
        makes params require grad
        '''
        pass

    def eval(self):
        '''
        makes params require no grad
        '''
        pass

    def to(self, device):
        '''
        moves parameters to device
        '''
        pass