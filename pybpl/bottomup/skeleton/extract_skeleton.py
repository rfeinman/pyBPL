import numpy as np
from skimage import morphology

from .sknw import build_sknw



def extract_skeleton(image):
    # get thinned image (skeleton)
    image = morphology.remove_small_holes(image, 2)
    image = morphology.binary_erosion(image)
    image = morphology.thin(image)
    # convert skeleton into network of junction nodes (undirected graph)
    graph = build_sknw(image.astype(np.uint16), multi=True)

    return graph