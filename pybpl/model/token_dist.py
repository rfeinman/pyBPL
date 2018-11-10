class TokenDist(object):
    '''
    prior over Token
    '''
    def __init__(self,lib):
        pass
    def sample_token(self,_type):
        '''
        Note: Should only be called from Model
        Note: Should return Token object
        '''
        pass
    def score_token(self,token):
        '''
        Note: Should only be called from Model
        Note: Should return a log probability
        '''
        pass