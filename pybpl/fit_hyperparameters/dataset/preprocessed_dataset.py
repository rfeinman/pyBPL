class PreprocessedDataset(object):
    def __init__(self, splines, drawings, scales):
        '''
        Use this on the output of omniglot_extract_splines.m

        import numpy as np
        from scipy.io import loadmat
        data = loadmat(
            'data_background_splines',
            variable_names=['bspline_substks','pdrawings_norm','pdrawings_scales']
        )
        D = PreprocessedDataset(
            data['bspline_substks'],
            data['pdrawings_norm'],
            data['pdrawings_scales']
        )
        '''
        self.splines = {}
        self.drawings = {}
        self.scales = {}

        n_alpha = len(splines)
        for a in range(n_alpha):
            alphabet_sp = splines[a, 0]
            alphabet_d = drawings[a, 0]
            alphabet_s = scales[a, 0]
            n_char = len(alphabet_sp)
            self.splines[a] = {}
            self.drawings[a] = {}
            self.scales[a] = {}
            for c in range(n_char):
                character_sp = alphabet_sp[c, 0]
                character_d = alphabet_d[c, 0]
                character_s = alphabet_s[c, 0]
                n_rend = len(character_sp)
                self.splines[a][c] = {}
                self.drawings[a][c] = {}
                self.scales[a][c] = {}
                for r in range(n_rend):
                    rendition_sp = character_sp[r, 0]
                    rendition_d = character_d[r, 0]
                    rendition_s = character_s[r, 0]
                    num_strokes = len(rendition_sp)
                    self.splines[a][c][r] = {}
                    self.drawings[a][c][r] = {}
                    self.scales[a][c][r] = {}
                    for s in range(num_strokes):
                        if rendition_sp[s, 0].size == 0:
                            continue
                        stroke_sp = rendition_sp[s, 0]
                        stroke_d = rendition_d[s, 0]
                        stroke_s = rendition_s[s, 0]
                        num_substrokes = len(stroke_sp)
                        self.splines[a][c][r][s] = {}
                        self.drawings[a][c][r][s] = {}
                        self.scales[a][c][r][s] = {}
                        for ss in range(num_substrokes):
                            self.splines[a][c][r][s][ss] = stroke_sp[ss, 0]
                            self.drawings[a][c][r][s][ss] = stroke_d[ss, 0]
                            self.scales[a][c][r][s][ss] = stroke_s[ss, 0]