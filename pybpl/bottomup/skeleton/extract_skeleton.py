import numpy as np
from skimage import morphology

from .sknw import build_sknw



def extract_skeleton(I):
    # get thinned image (skeleton)
    I = morphology.remove_small_holes(I, 2)
    I = morphology.thin(I)
    # convert skeleton into network of junction nodes (undirected graph)
    graph = build_sknw(I.astype(np.uint16))

    return graph