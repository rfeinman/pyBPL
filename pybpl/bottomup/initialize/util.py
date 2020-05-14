def fit_smooth_stk(stk, image, ps):
    raise NotImplementedError

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

def stroke_from_nodes(graph, list_ni):
    """
    Compute stroke trajectory from a graph and a list of nodes
    """
    raise NotImplementedError