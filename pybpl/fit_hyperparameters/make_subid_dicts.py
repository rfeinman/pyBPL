import torch
from .get_primitive_IDs import PrimitiveClassifierSingle

def make_subid_dict(ss_dict, spline_dict):
    clf = PrimitiveClassifierSingle(lib_dir='../../lib_data')
    subid_dict = {}

    n_alpha = len(ss_dict)
    for a in range(n_alpha):
        subid_dict[a] = {}
        alphabet = ss_dict[a]
        n_char = len(alphabet)
        for c in range(n_char):
            subid_dict[a][c] = {}
            char = alphabet[c]
            n_rend = len(char)
            for r in range(n_rend):
                subid_dict[a][c][r] = {}
                rendition = char[r]
                n_stroke = len(rendition)
                for s in range(n_stroke):

                    ids = []
                    stroke = rendition[s]
                    n_substrokes = len(stroke)
                    for ss in range(n_substrokes):
                        num_steps = len(stroke[ss])
                        if num_steps >= 10:
                            spline = torch.tensor(spline_dict[a][c][r][s][ss],
                                                  dtype=torch.float32)
                            prim_ID = clf.predict(spline)
                            ids.append(prim_ID)
                    subid_dict[a][c][r][s] = ids

    return subid_dict